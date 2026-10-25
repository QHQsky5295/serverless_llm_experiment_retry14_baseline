/* D844 diagnostic only: fresh lstat calls in one foreign-function boundary.
 * No directory-entry cache, cross-call state, worker thread, or serving import.
 * Compile on the measured Linux host; the public record uses fixed-width fields.
 */
#define _DEFAULT_SOURCE
#include <sys/stat.h>
#include <stdint.h>
#include <stddef.h>
#include <errno.h>
#include <dirent.h>
#include <stdlib.h>
#include <string.h>

struct ieee_file_stat {
    uint64_t device, inode, mode, nlink;
    int64_t size, blocks, mtime_sec, mtime_nsec, ctime_sec, ctime_nsec;
};

int ieee_file_stat_batch(const char *const *paths, size_t count,
                        struct ieee_file_stat *output, size_t *completed) {
    if (!completed) return EINVAL;
    *completed = 0;
    if (count && (!paths || !output)) return EINVAL;
    for (size_t i = 0; i < count; ++i) {
        if (!paths[i]) return EINVAL;
        struct stat value;
        if (lstat(paths[i], &value)) return errno;
        output[i] = (struct ieee_file_stat) {
            value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
            value.st_size, value.st_blocks, value.st_mtim.tv_sec,
            value.st_mtim.tv_nsec, value.st_ctim.tv_sec, value.st_ctim.tv_nsec
        };
        *completed = i + 1;
    }
    return 0;
}

size_t ieee_file_stat_record_bytes(void) { return sizeof(struct ieee_file_stat); }

/* D856 diagnostic: one transaction owns the first fresh observations. A second
 * fresh lstat validates them natively; no second metadata vector crosses into
 * Python. Nothing survives this transaction, and no serving code imports it.
 */
struct ieee_file_tree_entry {
    char *path;
    size_t root;
    struct ieee_file_stat value;
};

struct ieee_file_tree {
    struct ieee_file_tree_entry *entries;
    size_t count, capacity;
};

static struct ieee_file_stat record_of(struct stat value) {
    return (struct ieee_file_stat) {
        value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
        value.st_size, value.st_blocks, value.st_mtim.tv_sec,
        value.st_mtim.tv_nsec, value.st_ctim.tv_sec, value.st_ctim.tv_nsec
    };
}

void ieee_file_tree_free(struct ieee_file_tree *tree) {
    if (!tree) return;
    for (size_t i = 0; i < tree->count; ++i) free(tree->entries[i].path);
    free(tree->entries);
    free(tree);
}

static int append_path(struct ieee_file_tree *tree, char *path, size_t root) {
    if (tree->count == tree->capacity) {
        size_t capacity = tree->capacity ? tree->capacity * 2 : 16;
        if (capacity < tree->capacity || capacity > SIZE_MAX / sizeof(*tree->entries)) {
            free(path);
            return EOVERFLOW;
        }
        void *memory = realloc(tree->entries, capacity * sizeof(*tree->entries));
        if (!memory) { free(path); return ENOMEM; }
        tree->entries = memory;
        tree->capacity = capacity;
    }
    struct ieee_file_tree_entry *entry = &tree->entries[tree->count++];
    *entry = (struct ieee_file_tree_entry) {.path = path, .root = root};
    struct stat value;
    if (lstat(path, &value)) return errno;
    entry->value = record_of(value);
    return 0;
}

int ieee_file_tree_scan(const char *const *roots, size_t count,
                       struct ieee_file_tree **output, size_t *failed) {
    if (!output || !failed || (count && !roots)) return EINVAL;
    *output = NULL;
    *failed = 0;
    struct ieee_file_tree *tree = calloc(1, sizeof(*tree));
    if (!tree) return ENOMEM;
    *output = tree;  /* Caller owns cleanup even on failure. */
    for (size_t root = 0; root < count; ++root) {
        if (!roots[root]) return EINVAL;
        char *path = strdup(roots[root]);
        if (!path) return ENOMEM;
        *failed = tree->count;
        int status = append_path(tree, path, root);
        if (status) return status;
    }
    for (size_t i = 0; i < tree->count; ++i) {
        *failed = i;
        if (!S_ISDIR(tree->entries[i].value.mode)) continue;
        const char *parent = tree->entries[i].path;
        size_t root = tree->entries[i].root;
        DIR *directory = opendir(parent);
        if (!directory) return errno;
        int status = 0;
        for (;;) {
            errno = 0;
            struct dirent *child = readdir(directory);
            if (!child) { status = errno; break; }
            if (!strcmp(child->d_name, ".") || !strcmp(child->d_name, "..")) continue;
            size_t left = !strcmp(parent, ".") ? 0 : strlen(parent);
            size_t right = strlen(child->d_name);
            int separator = left && parent[left - 1] != '/';
            if (left > SIZE_MAX - right - 2) { status = EOVERFLOW; break; }
            char *path = malloc(left + (size_t)separator + right + 1);
            if (!path) { status = ENOMEM; break; }
            memcpy(path, parent, left);
            if (separator) path[left] = '/';
            memcpy(path + left + separator, child->d_name, right + 1);
            *failed = tree->count;
            status = append_path(tree, path, root);
            if (status) break;
            *failed = i;
        }
        int closed = closedir(directory);
        if (status) return status;
        if (closed) return errno;
    }
    return 0;
}

size_t ieee_file_tree_count(const struct ieee_file_tree *tree) { return tree->count; }
const struct ieee_file_tree_entry *ieee_file_tree_entries(const struct ieee_file_tree *tree) {
    return tree->entries;
}
size_t ieee_file_tree_entry_bytes(void) { return sizeof(struct ieee_file_tree_entry); }

/* flags: bit 0 = charged allocation envelope, bit 1 = active fixed-size writer.
 * Order matches the original Python DFS projection. reason 1 is identity/content
 * mutation, reason 2 is allocation outside the current charged envelope.
 */
int ieee_file_tree_validate(const struct ieee_file_tree *tree, const size_t *order,
                           const unsigned char *flags, const uint64_t *lower,
                           const uint64_t *upper, size_t *completed, int *reason) {
    if (!tree || !completed || !reason ||
            (tree->count && (!order || !flags || !lower || !upper))) return EINVAL;
    *completed = 0;
    *reason = 0;
    for (size_t position = 0; position < tree->count; ++position) {
        size_t i = order[position];
        if (i >= tree->count || (flags[i] & ~3)) return EINVAL;
        const struct ieee_file_tree_entry *entry = &tree->entries[i];
        struct stat value;
        if (lstat(entry->path, &value)) return errno;
        struct ieee_file_stat now = record_of(value), old = entry->value;
        if ((flags[i] & 1) && (now.blocks < 0 ||
                (uint64_t)now.blocks > UINT64_MAX / 512 ||
                (uint64_t)now.blocks * 512 < lower[i] ||
                (uint64_t)now.blocks * 512 > upper[i])) {
            *reason = 2;
            return EAGAIN;
        }
        if (old.device != now.device || old.inode != now.inode ||
                old.mode != now.mode || old.size != now.size || old.nlink != now.nlink ||
                (!(flags[i] & 1) && old.blocks != now.blocks) ||
                (!(flags[i] & 2) && (old.mtime_sec != now.mtime_sec ||
                 old.mtime_nsec != now.mtime_nsec || old.ctime_sec != now.ctime_sec ||
                 old.ctime_nsec != now.ctime_nsec))) {
            *reason = 1;
            return EAGAIN;
        }
        *completed = position + 1;
    }
    return 0;
}

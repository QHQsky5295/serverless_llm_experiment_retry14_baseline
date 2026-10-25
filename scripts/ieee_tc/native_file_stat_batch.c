/* D844 diagnostic only: fresh lstat calls in one foreign-function boundary.
 * No directory-entry cache, cross-call state, worker thread, or serving import.
 * Compile on the measured Linux host; the public record uses fixed-width fields.
 */
#define _DEFAULT_SOURCE
#include <sys/stat.h>
#include <stdint.h>
#include <stddef.h>
#include <errno.h>

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

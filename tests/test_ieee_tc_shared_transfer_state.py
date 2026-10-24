"""Real Linux IPC tests; no backend/GPU, workload, or performance assertion."""
import asyncio
import fcntl
import multiprocessing as mp
import os
import threading
import unittest

from faaslora.scheduling.shared_transfer_state import (
    PressureOwnerUnavailable, SharedTransferStateOwner, SharedTransferStateReader)


def projection(ids, state='attached'):
    return dict(state=state, active_transfer_ids=sorted(ids))


def reader_process(connection, descriptor):
    reader = SharedTransferStateReader(descriptor)
    try:
        connection.send(('ready', os.getpid()))
        while True:
            command = connection.recv()
            if command == 'stop':
                return
            if command == 'snapshot':
                connection.send(reader.snapshot('native'))
            elif command == 'hold':
                with reader.transaction('native') as outer:
                    # An inner snapshot must not release the outer lock.
                    assert reader.snapshot('native') == outer
                    connection.send(('held', outer))
                    assert connection.recv() == 'release'
                connection.send('released')
            else:
                raise ValueError(command)
    finally:
        reader.close()
        connection.close()


def writer_process(connection):
    owner = SharedTransferStateOwner('physical-files')
    owner.publish(active_transfer_ids=['unresolved'],
                  owners={'native': projection(['unresolved'])})
    connection.send(owner.descriptor)
    assert connection.recv() == 'exit-without-finish'
    os._exit(0)  # Deliberate test-process loss; no invented publication/cleanup.


class SharedTransferStateTests(unittest.TestCase):
    def setUp(self):
        self.owner = SharedTransferStateOwner('physical-files')
        self.owner.publish(active_transfer_ids=[], owners={'native': projection([])})
        self.reader = SharedTransferStateReader(self.owner.descriptor)
        self.children = []

    def tearDown(self):
        for child, channel in self.children:
            if child.is_alive():
                try:
                    channel.send('stop')
                except (BrokenPipeError, EOFError):
                    pass
                child.join(5)
            if child.is_alive():
                child.terminate()  # Only a child created by this exact test.
                child.join(5)
            channel.close()
        self.reader.close()
        self.owner.publish(active_transfer_ids=[], owners={})
        self.owner.close()

    def start_reader(self):
        parent, child = mp.get_context('spawn').Pipe()
        process = mp.get_context('spawn').Process(
            target=reader_process, args=(child, self.owner.descriptor))
        process.start()
        child.close()
        self.children.append((process, parent))
        self.assertTrue(parent.poll(5), 'reader did not attach')
        state, pid = parent.recv()
        self.assertEqual(state, 'ready')
        self.assertNotEqual(pid, os.getpid())
        return process, parent

    def receive(self, pipe):
        self.assertTrue(pipe.poll(5), 'child did not complete its bounded command')
        return pipe.recv()

    def test_two_real_readers_see_one_physical_interval_without_event_rpcs(self):
        peers = [self.start_reader() for _ in range(2)]
        receipt = self.owner.publish(active_transfer_ids=['transfer'],
            owners={'native': projection(['transfer'])})
        self.assertFalse(receipt['native_acknowledgement'])
        for _, peer in peers:
            peer.send('snapshot')
            state = self.receive(peer)
            self.assertEqual(state['active_transfer_ids'], ['transfer'])
            self.assertEqual(state['transfer_sequence'], receipt['revision'])
            self.assertFalse(state['physical_capacity_reserved'])
        self.owner.publish(active_transfer_ids=[], owners={'native': projection([])})
        for _, peer in peers:
            peer.send('snapshot')
            self.assertEqual(self.receive(peer)['active_transfers'], 0)

    def test_native_observation_use_turn_excludes_publication_and_preserves_nested_lock(self):
        _, peer = self.start_reader()
        peer.send('hold')
        self.assertEqual(self.receive(peer)[0], 'held')
        # Writer fd and reader fd are distinct open descriptions, even though
        # they refer to the same memfd. The kernel must enforce exclusion.
        with self.assertRaises(BlockingIOError):
            fcntl.flock(self.owner._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        entered, finished = threading.Event(), threading.Event()
        errors = []
        def publish():
            entered.set()
            try:
                self.owner.publish(active_transfer_ids=['next'],
                                   owners={'native': projection(['next'])})
            except BaseException as exc:
                errors.append(exc)
            finally:
                finished.set()
        task = threading.Thread(target=publish)
        task.start()
        try:
            self.assertTrue(entered.wait(5))
            self.assertFalse(finished.wait(.05))
        finally:
            peer.send('release')
            self.assertEqual(self.receive(peer), 'released')
            task.join(5)
        self.assertFalse(task.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(self.reader.snapshot('native')['active_transfer_ids'], ['next'])

    def test_repeated_cancel_joins_real_publication_before_propagation(self):
        _, peer = self.start_reader()
        peer.send('hold')
        self.assertEqual(self.receive(peer)[0], 'held')
        started = threading.Event()
        original = self.owner.publish
        def announce(**kwargs):
            started.set()
            return original(**kwargs)
        self.owner.publish = announce
        async def run():
            task = asyncio.create_task(self.owner.publish_async(active_transfer_ids=['cancelled'],
                owners={'native': projection(['cancelled'])}))
            self.assertTrue(await asyncio.to_thread(started.wait, 5))
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            peer.send('release')
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)
            self.assertEqual(self.receive(peer), 'released')
        asyncio.run(run())
        self.assertEqual(self.reader.snapshot('native')['active_transfer_ids'], ['cancelled'])
        # Cancellation has not forged a finish; the caller still owns cleanup.
        with self.assertRaisesRegex(PressureOwnerUnavailable, 'unresolved'):
            self.owner.close()

    def test_attach_during_io_and_retirement_project_only_owned_intervals(self):
        self.owner.publish(active_transfer_ids=['old'], owners={'native': projection(['old'])})
        late = SharedTransferStateReader(self.owner.descriptor)
        try:
            with self.assertRaisesRegex(PressureOwnerUnavailable, 'subscription'):
                late.snapshot('late')
            self.owner.publish(active_transfer_ids=['old'], owners={
                'native': projection(['old']), 'late': projection(['old'])})
            self.assertEqual(late.snapshot('late')['active_transfer_ids'], ['old'])
            self.owner.publish(active_transfer_ids=['new', 'old'], owners={
                'native': projection(['old'], 'retiring'), 'late': projection(['new', 'old'])})
            self.assertEqual(self.reader.snapshot('native')['active_transfer_ids'], ['old'])
            self.assertEqual(late.snapshot('late')['active_transfer_ids'], ['new', 'old'])
            self.owner.publish(active_transfer_ids=['new'], owners={
                'native': projection([], 'retired'), 'late': projection(['new'])})
            with self.assertRaisesRegex(PressureOwnerUnavailable, 'subscription'):
                self.reader.snapshot('native')
        finally:
            late.close()

    def test_unresolved_interval_cannot_be_closed_as_idle(self):
        self.owner.publish(active_transfer_ids=['unresolved'],
                           owners={'native': projection(['unresolved'])})
        with self.assertRaisesRegex(PressureOwnerUnavailable, 'unresolved'):
            self.owner.close()
        self.assertEqual(self.reader.snapshot('native')['active_transfers'], 1)

    def test_owner_exit_invalidates_readable_nonempty_memfd(self):
        parent, child = mp.get_context('spawn').Pipe()
        process = mp.get_context('spawn').Process(target=writer_process, args=(child,))
        process.start()
        child.close()
        self.children.append((process, parent))
        descriptor = self.receive(parent)
        reader = SharedTransferStateReader(descriptor)
        try:
            self.assertEqual(reader.snapshot('native')['active_transfers'], 1)
            parent.send('exit-without-finish')
            process.join(5)
            self.assertEqual(process.exitcode, 0)
            self.assertGreater(os.fstat(reader._fd).st_size, 0)
            with self.assertRaisesRegex(PressureOwnerUnavailable, 'exited'):
                reader.snapshot('native')
        finally:
            reader.close()

    def test_no_reader_write_privilege_and_no_cross_thread_reuse(self):
        with self.assertRaises(OSError):
            os.write(self.reader._fd, b'changed')
        errors = []
        def other_thread():
            try:
                self.reader.snapshot('native')
            except BaseException as exc:
                errors.append(exc)
        thread = threading.Thread(target=other_thread)
        thread.start()
        thread.join(5)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], PressureOwnerUnavailable)

    def test_malformed_state_is_not_zero_pressure(self):
        fcntl.flock(self.owner._fd, fcntl.LOCK_EX)
        try:
            os.pwrite(self.owner._fd, b'!', 0)
        finally:
            fcntl.flock(self.owner._fd, fcntl.LOCK_UN)
        with self.assertRaisesRegex(PressureOwnerUnavailable, 'certify'):
            self.reader.snapshot('native')

    def test_wrong_inode_generation_and_clock_cannot_attach(self):
        for field, value in [('inode', self.owner.descriptor['inode'] + 1),
                             ('generation_id', 'other'), ('clock_id', 'other-clock')]:
            with self.subTest(field=field), self.assertRaises((ValueError, PressureOwnerUnavailable)):
                SharedTransferStateReader(dict(self.owner.descriptor, **{field: value}))

    def test_invalid_projection_never_replaces_valid_revision(self):
        before = self.reader.snapshot('native')
        for active, owners in [(['x', 'x'], {'native': projection(['x'])}),
                               (['x'], {'native': projection([])}),
                               ([], {'native': projection(['x'])}),
                               (['x'], {'native': projection(['x'], 'retired')})]:
            with self.subTest(active=active, owners=owners), self.assertRaises(ValueError):
                self.owner.publish(active_transfer_ids=active, owners=owners)
            self.assertEqual(self.reader.snapshot('native'), before)

    def test_explicit_close_is_unavailable_not_zero_for_existing_reader(self):
        owner = SharedTransferStateOwner('other-files')
        owner.publish(active_transfer_ids=[], owners={'native': projection([])})
        reader = SharedTransferStateReader(owner.descriptor)
        try:
            owner.close()
            with self.assertRaisesRegex(PressureOwnerUnavailable, 'certify'):
                reader.snapshot('native')
        finally:
            reader.close()

    def test_alternating_complete_revisions_with_spawned_readers(self):
        peers = [self.start_reader() for _ in range(2)]
        for i in range(30):
            ids = [f'transfer-{i}'] if i % 2 else []
            revision = self.owner.publish(active_transfer_ids=ids,
                owners={'native': projection(ids)})['revision']
            for _, peer in peers:
                peer.send('snapshot')
            for _, peer in peers:
                value = self.receive(peer)
                self.assertEqual(value['transfer_sequence'], revision)
                self.assertEqual(value['active_transfer_ids'], ids)


if __name__ == '__main__':
    unittest.main()

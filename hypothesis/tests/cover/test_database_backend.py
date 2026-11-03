# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import os
import re
import shutil
import tempfile
import time
import zipfile
from collections import Counter
from collections.abc import Iterable, Iterator
from contextlib import contextmanager, nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path
from shutil import make_archive, rmtree

import pytest

from hypothesis import (
    HealthCheck,
    configuration,
    example,
    given,
    settings,
    strategies as st,
)
from hypothesis.database import (
    BackgroundWriteDatabase,
    DirectoryBasedExampleDatabase,
    ExampleDatabase,
    GitHubArtifactDatabase,
    InMemoryExampleDatabase,
    MultiplexedDatabase,
    NamespacedDatabase,
    ReadOnlyDatabase,
    _db_for_path,
    _decode_key,
    _encode_key,
    _hash,
    _pack_uleb128,
    _unpack_uleb128,
    choices_from_bytes,
    choices_to_bytes,
)
from hypothesis.errors import (
    HypothesisDeprecationWarning,
    HypothesisWarning,
    InvalidArgument,
)
from hypothesis.internal.compat import WINDOWS
from hypothesis.internal.conjecture.choice import choice_equal
from hypothesis.stateful import (
    Bundle,
    RuleBasedStateMachine,
    invariant,
    precondition,
    rule,
    run_state_machine_as_test,
)
from hypothesis.strategies import binary, lists, tuples
from hypothesis.utils.conventions import not_set

from tests.common.utils import (
    checks_deprecated_behaviour,
    skipif_emscripten,
    skipif_root,
    skipif_threading,
    wait_for,
)
from tests.conjecture.common import nodes, nodes_inline

# we need real time here, not monkeypatched for CI
time_sleep = time.sleep


@given(lists(tuples(binary(), binary())))
@settings(max_examples=50)
def test_backend_returns_what_you_put_in(xs):
    backend = InMemoryExampleDatabase()
    mapping = {}
    for key, value in xs:
        mapping.setdefault(key, set()).add(value)
        backend.save(key, value)
    for key, values in mapping.items():
        backend_contents = list(backend.fetch(key))
        distinct_backend_contents = set(backend_contents)
        assert len(backend_contents) == len(distinct_backend_contents)
        assert distinct_backend_contents == set(values)


def test_can_delete_keys():
    backend = InMemoryExampleDatabase()
    backend.save(b"foo", b"bar")
    backend.save(b"foo", b"baz")
    backend.delete(b"foo", b"bar")
    assert list(backend.fetch(b"foo")) == [b"baz"]


def test_default_database_is_in_memory():
    with pytest.warns(HypothesisDeprecationWarning):
        assert isinstance(ExampleDatabase(), InMemoryExampleDatabase)


def test_default_on_disk_database_is_dir(tmp_path):
    with pytest.warns(HypothesisDeprecationWarning):
        assert isinstance(
            ExampleDatabase(tmp_path.joinpath("foo")), DirectoryBasedExampleDatabase
        )


def test_does_not_error_when_fetching_when_not_exist(tmp_path):
    db = DirectoryBasedExampleDatabase(tmp_path / "examples")
    db.fetch(b"foo")


@pytest.fixture(scope="function", params=["memory", "directory"])
def exampledatabase(request, tmp_path):
    if request.param == "memory":
        return InMemoryExampleDatabase()
    assert request.param == "directory"
    return DirectoryBasedExampleDatabase(tmp_path / "examples")


def test_can_delete_a_key_that_is_not_present(exampledatabase):
    exampledatabase.delete(b"foo", b"bar")


def test_can_fetch_a_key_that_is_not_present(exampledatabase):
    assert list(exampledatabase.fetch(b"foo")) == []


def test_saving_a_key_twice_fetches_it_once(exampledatabase):
    exampledatabase.save(b"foo", b"bar")
    exampledatabase.save(b"foo", b"bar")
    assert list(exampledatabase.fetch(b"foo")) == [b"bar"]


def test_can_close_a_database_after_saving(exampledatabase):
    exampledatabase.save(b"foo", b"bar")


def test_class_name_is_in_repr(exampledatabase):
    assert type(exampledatabase).__name__ in repr(exampledatabase)


def test_an_absent_value_is_present_after_it_moves(exampledatabase):
    exampledatabase.move(b"a", b"b", b"c")
    assert next(exampledatabase.fetch(b"b")) == b"c"


def test_an_absent_value_is_present_after_it_moves_to_self(exampledatabase):
    exampledatabase.move(b"a", b"a", b"b")
    assert next(exampledatabase.fetch(b"a")) == b"b"


@skipif_threading
def test_two_directory_databases_can_interact(tmp_path):
    db1 = DirectoryBasedExampleDatabase(tmp_path)
    db2 = DirectoryBasedExampleDatabase(tmp_path)
    db1.save(b"foo", b"bar")
    assert list(db2.fetch(b"foo")) == [b"bar"]
    db2.save(b"foo", b"bar")
    db2.save(b"foo", b"baz")
    assert sorted(db1.fetch(b"foo")) == [b"bar", b"baz"]


def test_can_handle_disappearing_files(tmp_path, monkeypatch):
    db = DirectoryBasedExampleDatabase(tmp_path)
    db.save(b"foo", b"bar")
    base_listdir = os.listdir
    monkeypatch.setattr(
        os, "listdir", lambda d: [*base_listdir(d), "this-does-not-exist"]
    )
    assert list(db.fetch(b"foo")) == [b"bar"]


def test_readonly_db_is_not_writable():
    inner = InMemoryExampleDatabase()
    wrapped = ReadOnlyDatabase(inner)
    inner.save(b"key", b"value")
    inner.save(b"key", b"value2")
    wrapped.delete(b"key", b"value")
    wrapped.move(b"key", b"key2", b"value2")
    wrapped.save(b"key", b"value3")
    assert set(wrapped.fetch(b"key")) == {b"value", b"value2"}
    assert set(wrapped.fetch(b"key2")) == set()


def test_multiplexed_dbs_read_and_write_all():
    a = InMemoryExampleDatabase()
    b = InMemoryExampleDatabase()
    multi = MultiplexedDatabase(a, b)
    a.save(b"a", b"aa")
    b.save(b"b", b"bb")
    multi.save(b"c", b"cc")
    multi.move(b"a", b"b", b"aa")
    for db in (a, b, multi):
        assert set(db.fetch(b"a")) == set()
        assert set(db.fetch(b"c")) == {b"cc"}
    got = list(multi.fetch(b"b"))
    assert len(got) == 2
    assert set(got) == {b"aa", b"bb"}
    multi.delete(b"c", b"cc")
    for db in (a, b, multi):
        assert set(db.fetch(b"c")) == set()


def test_ga_require_readonly_wrapping():
    """Test that GitHubArtifactDatabase requires wrapping around ReadOnlyDatabase"""
    database = GitHubArtifactDatabase("test", "test")
    # save, move and delete can only be called when wrapped around ReadonlyDatabase
    with pytest.raises(RuntimeError, match=re.escape(database._read_only_message)):
        database.save(b"foo", b"bar")
    with pytest.raises(RuntimeError):
        database.move(b"foo", b"bar", b"foobar")
    with pytest.raises(RuntimeError):
        database.delete(b"foo", b"bar")

    # check that the database silently ignores writes when wrapped around ReadOnlyDatabase
    database = ReadOnlyDatabase(database)
    database.save(b"foo", b"bar")
    database.move(b"foo", b"bar", b"foobar")
    database.delete(b"foo", b"bar")


@contextmanager
def ga_empty_artifact(
    date: datetime | None = None, path: Path | None = None
) -> Iterator[tuple[Path, Path]]:
    """Creates an empty GitHub artifact."""
    if date:
        timestamp = date.isoformat().replace(":", "_")
    else:
        timestamp = datetime.now(timezone.utc).isoformat().replace(":", "_")

    temp_dir = None
    if not path:
        temp_dir = tempfile.mkdtemp()
        path = Path(temp_dir) / "github-artifacts"

    path.mkdir(parents=True, exist_ok=True)
    zip_path = path / f"{timestamp}.zip"

    with zipfile.ZipFile(zip_path, "w"):
        pass

    try:
        yield (path, zip_path)
    finally:
        if temp_dir:
            rmtree(temp_dir)


def test_ga_empty_read():
    """Tests that an inexistent key returns nothing."""
    with ga_empty_artifact() as (path, _):
        database = GitHubArtifactDatabase("test", "test", path=path)
        assert list(database.fetch(b"foo")) == []


def test_ga_initialize():
    """
    Tests that the database is initialized when a new artifact is found.
    As well that initialization doesn't happen again on the next fetch.
    """
    now = datetime.now(timezone.utc)
    with ga_empty_artifact(date=(now - timedelta(hours=2))) as (path, _):
        database = GitHubArtifactDatabase("test", "test", path=path)
        # Trigger initialization
        list(database.fetch(b""))
        initial_artifact = database._artifact
        assert initial_artifact
        assert database._artifact
        assert database._access_cache is not None
        with ga_empty_artifact(date=now, path=path) as (path, _):
            # Initialization shouldn't happen again
            list(database.fetch(b""))
            assert database._initialized
            assert database._artifact == initial_artifact


def test_ga_no_artifact(tmp_path):
    """Tests that the database is disabled when no artifact is found."""
    database = GitHubArtifactDatabase("test", "test", path=tmp_path)
    # Check that the database raises a warning
    with pytest.warns(HypothesisWarning):
        assert list(database.fetch(b"")) == []
    assert database._disabled is True
    assert list(database.fetch(b"")) == []


def test_ga_corrupted_artifact():
    """Tests that corrupted artifacts are properly detected and warned about."""
    with ga_empty_artifact() as (path, zip_path):
        # Corrupt the CRC of the zip file
        with open(zip_path, "rb+") as f:
            f.write(b"\x00\x01\x00\x01")

        database = GitHubArtifactDatabase("test", "test", path=path)
        # Check that the database raises a warning
        with pytest.warns(HypothesisWarning):
            assert list(database.fetch(b"")) == []
        assert database._disabled is True


def test_ga_reads_artifact_without_directory_entries():
    # see https://github.com/HypothesisWorks/hypothesis/pull/4787
    key = b"foo"
    value = b"bar"
    with ga_empty_artifact() as (path, zip_path):
        with zipfile.ZipFile(zip_path, "w") as zf:
            # write only a file entry, without a directory entry for its parent
            zf.writestr(f"{_hash(key)}/{_hash(value)}", value)

        database = GitHubArtifactDatabase("test", "test", path=path)
        assert list(database.fetch(key)) == [value]


def test_ga_deletes_old_artifacts():
    """Tests that old artifacts are automatically deleted."""
    now = datetime.now(timezone.utc)
    with (
        ga_empty_artifact(date=now) as (path, file_now),
        ga_empty_artifact(date=now - timedelta(hours=2), path=path) as (
            _,
            file_old,
        ),
    ):
        database = GitHubArtifactDatabase("test", "test", path=path)
        # Trigger initialization
        list(database.fetch(b""))
        assert file_now.exists()
        assert not file_old.exists()


@skipif_threading
def test_ga_triggers_fetching(monkeypatch, tmp_path):
    """Tests whether an artifact fetch is triggered, and an expired artifact is deleted."""
    with ga_empty_artifact() as (_, artifact):
        # We patch the _fetch_artifact method to return our artifact
        def fake_fetch_artifact(self) -> Path | None:
            return artifact

        monkeypatch.setattr(
            GitHubArtifactDatabase, "_fetch_artifact", fake_fetch_artifact
        )

        database = GitHubArtifactDatabase(
            "test", "test", path=tmp_path, cache_timeout=timedelta(days=1)
        )

        # Test without an existing artifact
        list(database.fetch(b""))

        assert not database._disabled
        assert database._initialized
        assert database._artifact == artifact

        # Now we'll see if the DB also fetched correctly with an expired artifact
        now = datetime.now(timezone.utc)
        # We create an expired artifact
        with ga_empty_artifact(date=now - timedelta(days=2)) as (
            path_with_artifact,
            old_artifact,
        ):
            database = GitHubArtifactDatabase(
                "test", "test", path=path_with_artifact, cache_timeout=timedelta(days=1)
            )

            # Trigger initialization
            list(database.fetch(b""))
            assert not database._disabled
            assert database._initialized
            assert database._artifact == artifact

            # Check that the artifact was deleted
            assert not old_artifact.exists()


def test_ga_fallback_expired(monkeypatch):
    """
    Tests that the fallback to an expired artifact is triggered
    if fetching a new one fails. This allows for (by example) offline development.
    """
    now = datetime.now(timezone.utc)
    with ga_empty_artifact(date=now - timedelta(days=2)) as (path, artifact):
        database = GitHubArtifactDatabase(
            "test", "test", path=path, cache_timeout=timedelta(days=1)
        )

        # This should trigger the fallback
        def fake_fetch_artifact(self) -> Path | None:
            return None

        monkeypatch.setattr(
            GitHubArtifactDatabase, "_fetch_artifact", fake_fetch_artifact
        )

        # Trigger initialization
        with pytest.warns(HypothesisWarning):
            list(database.fetch(b""))

        assert not database._disabled
        assert database._initialized
        assert database._artifact == artifact


class GitHubArtifactMocks(RuleBasedStateMachine):
    """
    This is a state machine that tests agreement of GitHubArtifactDatabase
    with DirectoryBasedExampleDatabase (as a reference implementation).
    """

    def __init__(self):
        super().__init__()
        self.temp_directory = Path(tempfile.mkdtemp())
        self.path = self.temp_directory / "github-artifacts"

        # This is where we will store the contents for the zip file
        timestamp = datetime.now(timezone.utc).isoformat().replace(":", "_")
        self.zip_destination = self.path / f"{timestamp}.zip"

        # And this is where we want to create it
        self.zip_content_path = self.path / timestamp
        self.zip_content_path.mkdir(parents=True, exist_ok=True)

        # We use a DirectoryBasedExampleDatabase to create the contents
        self.directory_db = DirectoryBasedExampleDatabase(str(self.zip_content_path))
        self.zip_db = GitHubArtifactDatabase("mock", "mock", path=self.path)

        # Create zip file for the first time
        self._archive_directory_db()
        self.zip_db._initialize_db()

    def _make_zip(self, tree_path: Path, zip_path: Path):
        destination = zip_path.parent.absolute() / zip_path.stem
        make_archive(
            str(destination),
            "zip",
            root_dir=tree_path,
        )

    def _archive_directory_db(self):
        # Delete all of the zip files in the directory
        for file in self.path.glob("*.zip"):
            file.unlink()

        self._make_zip(self.zip_content_path, self.zip_destination)

    keys = Bundle("keys")
    values = Bundle("values")

    @rule(target=keys, k=st.binary())
    def k(self, k):
        return k

    @rule(target=values, v=st.binary())
    def v(self, v):
        return v

    @rule(k=keys, v=values)
    def save(self, k, v):
        self.directory_db.save(k, v)
        self._archive_directory_db()
        self.zip_db = GitHubArtifactDatabase("mock", "mock", path=self.path)
        self.zip_db._initialize_db()

    @rule(k=keys)
    def values_agree(self, k):
        v1 = set(self.directory_db.fetch(k))
        v2 = set(self.zip_db.fetch(k))

        assert v1 == v2

    def teardown(self):
        shutil.rmtree(self.temp_directory)


TestGADReads = GitHubArtifactMocks.TestCase


def test_gadb_coverage():
    # Ensure that we always cover the nonempty-archive case, which can otherwise
    # cause rare incomplete-coverage failures.
    state = GitHubArtifactMocks()
    state.save(b"key", b"value")
    state.values_agree(b"key")


@skipif_root
@pytest.mark.parametrize("dirs", [[], ["subdir"]])
def test_database_directory_inaccessible(dirs, tmp_path, monkeypatch):
    monkeypatch.setattr(
        configuration, "__hypothesis_home_directory", tmp_path.joinpath(*dirs)
    )
    try:
        tmp_path.chmod(0o000)
        with (
            nullcontext()
            if WINDOWS
            else pytest.warns(
                HypothesisWarning, match=".*the default location is unusable"
            )
        ):
            database = _db_for_path(not_set)
        database.save(b"fizz", b"buzz")
    finally:
        tmp_path.chmod(0o600)  # So that pytest can clean up tmp_path later


@skipif_emscripten
def test_background_write_database():
    db = BackgroundWriteDatabase(InMemoryExampleDatabase())
    db.save(b"a", b"b")
    db.save(b"a", b"c")
    db.save(b"a", b"d")
    assert set(db.fetch(b"a")) == {b"b", b"c", b"d"}

    db.move(b"a", b"a2", b"b")
    assert set(db.fetch(b"a")) == {b"c", b"d"}
    assert set(db.fetch(b"a2")) == {b"b"}

    db.delete(b"a", b"c")
    assert set(db.fetch(b"a")) == {b"d"}


@given(lists(nodes()))
# covering examples
@example(nodes_inline(True))
@example(nodes_inline(1))
@example(nodes_inline(0.0))
@example(nodes_inline(-0.0))
@example(nodes_inline("a"))
@example(nodes_inline(b"a"))
@example(nodes_inline(b"a" * 50))
@example(nodes_inline(b"1" * 100_000))  # really long bytes
def test_nodes_roundtrips(nodes1):
    s1 = choices_to_bytes([n.value for n in nodes1])
    assert isinstance(s1, bytes)
    ir2 = choices_from_bytes(s1)
    assert len(nodes1) == len(ir2)

    for n1, v2 in zip(nodes1, ir2, strict=True):
        assert choice_equal(n1.value, v2)

    s2 = choices_to_bytes(ir2)
    assert s1 == s2


@given(st.integers(min_value=0))
def test_uleb_128_roundtrips(n1):
    buffer1 = _pack_uleb128(n1)
    idx, n2 = _unpack_uleb128(buffer1)
    assert idx == len(buffer1)
    assert n1 == n2


def _database_conforms_to_listener_api(
    create_db,
    *,
    flush=None,
    supports_value_delete=True,
    parent_settings=None,
):
    # this function is a big mess to support a bunch of different special cases
    # for different databases, sorry. In return, we get one big stateful test
    # we can use to test the listener api for all of our databases.
    #
    # * create_db is a callable which accepts one argument (a path to a temporary
    #   directory) and returns a database instance.
    # * flush is a callable which takes the instantiated db as an argument, and
    #   is called on every step as an invariant. This lets the database do things
    #   like, time.sleep to give time for events to fire.
    # * suports_value_delete is True if the db supports passing
    #   the exact value of a deleted key in "delete" events. The directory database
    #   notably does not support this, and passes None instead.

    @settings(parent_settings, suppress_health_check=[HealthCheck.too_slow])
    class TestDatabaseListener(RuleBasedStateMachine):
        # this tests that if we call .delete, .save, or .move in a database, and
        # that operation changes the state of the database, any registered listeners
        # get called a corresponding number of times.
        keys = Bundle("keys")
        values = Bundle("values")

        def __init__(self):
            super().__init__()

            self.temp_dir = Path(tempfile.mkdtemp())
            self.db = create_db(self.temp_dir)
            self.expected_events = []
            self.actual_events = []

            def listener(event):
                self.actual_events.append(event)

            self.listener = listener
            self.active_listeners = []
            self.add_listener()

        def _expect_event(self, event_type, args):
            for _ in range(len(self.active_listeners)):
                self.expected_events.append((event_type, args))

        def _expect_delete(self, k, v):
            if not supports_value_delete:
                v = None
            self._expect_event("delete", (k, v))

        def _expect_save(self, k, v):
            self._expect_event("save", (k, v))

        @rule(target=keys, k=st.binary())
        def k(self, k):
            return k

        @rule(target=values, v=st.binary())
        def v(self, v):
            return v

        @precondition(lambda self: not self.active_listeners)
        @rule()
        def add_listener(self):
            self.db.add_listener(self.listener)
            # wait for watchdog to initialize the listener asynchronously
            time_sleep(0.1)
            self.active_listeners.append(self.listener)

        @precondition(lambda self: self.listener in self.active_listeners)
        @rule()
        def remove_listener(self):
            self.db.remove_listener(self.listener)
            self.active_listeners.remove(self.listener)

        @rule()
        def clear_listeners(self):
            self.db.clear_listeners()
            self.active_listeners.clear()

        @rule(k=keys)
        def fetch(self, k):
            # we don't expect this to do anything, but that's the point. if this
            # fires a listener call then that's bad and will fail.
            self.db.fetch(k)

        @rule(k=keys, v=values)
        def save(self, k, v):
            changed = v not in set(self.db.fetch(k))
            self.db.save(k, v)

            if changed:
                self._expect_save(k, v)

        @rule(k=keys, v=values)
        def delete(self, k, v):
            changed = v in set(self.db.fetch(k))
            self.db.delete(k, v)

            if changed:
                self._expect_delete(k, v)

        @rule(k1=keys, k2=keys, v=values)
        def move(self, k1, k2, v):
            in_k1 = v in set(self.db.fetch(k1))
            save_changed = v not in set(self.db.fetch(k2))
            delete_changed = k1 != k2 and in_k1
            self.db.move(k1, k2, v)

            # A move gets emitted as a delete followed by a save.  The
            # delete may be omitted if k1==k2, and the save if v in db.fetch(k2).
            if delete_changed:
                self._expect_delete(k1, v)
            if save_changed:
                self._expect_save(k2, v)

        # it would be nice if this was an @rule, but that runs into race condition
        # failures where an event listener is removed immediately after a
        # save/delete/move operation, before the listener can fire. This is only
        # relevant for DirectoryBasedExampleDatabase.
        @invariant()
        def events_agree(self):
            if flush is not None:
                flush(self.db)

            wait_for(
                lambda: len(self.expected_events) == len(self.actual_events), timeout=60
            )
            # events *generally* don't arrive out of order, but we've had
            # flakes reported here, especially on weirder / older machines.
            # see https://github.com/HypothesisWorks/hypothesis/issues/4274
            assert Counter(self.expected_events) == Counter(self.actual_events)

        def teardown(self):
            shutil.rmtree(self.temp_dir)

    run_state_machine_as_test(TestDatabaseListener)


def test_database_listener_memory():
    _database_conforms_to_listener_api(
        lambda path: InMemoryExampleDatabase(),
        parent_settings=settings(max_examples=5, stateful_step_count=10),
    )


@skipif_emscripten
@pytest.mark.skipif(
    settings.get_current_profile_name() == "crosshair", reason="takes ages"
)
def test_database_listener_background_write():
    _database_conforms_to_listener_api(
        lambda path: BackgroundWriteDatabase(InMemoryExampleDatabase()),
        flush=lambda db: db._join(),
        parent_settings=settings(max_examples=5, stateful_step_count=10),
    )


def test_can_remove_nonexistent_listener():
    db = InMemoryExampleDatabase()
    db.remove_listener(lambda event: event)


class DoesNotSupportListening(ExampleDatabase):
    def save(self, key: bytes, value: bytes) -> None: ...
    def fetch(self, key: bytes) -> Iterable[bytes]: ...
    def delete(self, key: bytes, value: bytes) -> None: ...


def test_warns_when_listening_not_supported():
    db = DoesNotSupportListening()
    listener = lambda event: event

    with pytest.warns(
        HypothesisWarning, match="does not support listening for changes"
    ):
        db.add_listener(listener)

    with pytest.warns(
        HypothesisWarning, match="does not support stopping listening for changes"
    ):
        db.remove_listener(listener)


def test_readonly_listener():
    db = ReadOnlyDatabase(InMemoryExampleDatabase())

    def listener(event):
        raise AssertionError("ReadOnlyDatabase never fires change events")

    db.add_listener(listener)
    db.save(b"a", b"a")

    db.remove_listener(listener)
    db.save(b"b", b"b")


@skipif_threading
def test_metakeys_move_into_existing_key(tmp_path):
    db = DirectoryBasedExampleDatabase(tmp_path)
    db.save(b"k1", b"v1")
    db.save(b"k1", b"v2")
    db.save(b"k2", b"v3")
    assert set(db.fetch(db._metakeys_name)) == {b"k1", b"k2"}

    db.move(b"k1", b"k2", b"v2")
    assert set(db.fetch(db._metakeys_name)) == {b"k1", b"k2"}


@skipif_threading
def test_metakeys_move_into_nonexistent_key(tmp_path):
    db = DirectoryBasedExampleDatabase(tmp_path)
    db.save(b"k1", b"v1")
    assert set(db.fetch(db._metakeys_name)) == {b"k1"}

    db.move(b"k1", b"k2", b"v1")
    assert set(db.fetch(db._metakeys_name)) == {b"k1", b"k2"}


@skipif_threading
def test_metakeys(tmp_path):
    db = DirectoryBasedExampleDatabase(tmp_path)

    db.save(b"k1", b"v1")
    assert set(db.fetch(db._metakeys_name)) == {b"k1"}

    db.save(b"k1", b"v2")
    assert set(db.fetch(db._metakeys_name)) == {b"k1"}

    # deleting all the values from a key removes that metakey
    db.delete(b"k1", b"v1")
    db.delete(b"k1", b"v2")
    assert set(db.fetch(db._metakeys_name)) == set()

    db.save(b"k2", b"v1")
    assert set(db.fetch(db._metakeys_name)) == {b"k2"}


class TracksListens(ExampleDatabase):
    def __init__(self):
        super().__init__()
        self.starts = 0
        self.ends = 0

    def save(self, key: bytes, value: bytes) -> None: ...
    def fetch(self, key: bytes) -> Iterable[bytes]: ...
    def delete(self, key: bytes, value: bytes) -> None: ...

    def _start_listening(self):
        self.starts += 1

    def _stop_listening(self):
        self.ends += 1


def test_start_end_listening():
    db = TracksListens()

    def listener1(event):
        pass

    def listener2(event):
        pass

    assert db.starts == 0
    db.add_listener(listener1)
    assert db.starts == 1
    db.add_listener(listener2)
    assert db.starts == 1

    assert db.ends == 0
    db.remove_listener(listener2)
    assert db.ends == 0
    db.remove_listener(listener1)
    assert db.ends == 1

    db.clear_listeners()
    assert db.ends == 1


@checks_deprecated_behaviour
def test_deprecated_example_database_path(tmp_path):
    ExampleDatabase(tmp_path)


@checks_deprecated_behaviour
def test_deprecated_example_database_memory():
    ExampleDatabase(":memory:")


@checks_deprecated_behaviour
def test_deprecated_example_database_no_args():
    ExampleDatabase()


@pytest.mark.parametrize(
    "db1, db2",
    [
        (DirectoryBasedExampleDatabase("a"), DirectoryBasedExampleDatabase("a")),
        (
            MultiplexedDatabase(
                DirectoryBasedExampleDatabase("a"), DirectoryBasedExampleDatabase("b")
            ),
            MultiplexedDatabase(
                DirectoryBasedExampleDatabase("a"), DirectoryBasedExampleDatabase("b")
            ),
        ),
        (
            ReadOnlyDatabase(DirectoryBasedExampleDatabase("a")),
            ReadOnlyDatabase(DirectoryBasedExampleDatabase("a")),
        ),
        (
            NamespacedDatabase(DirectoryBasedExampleDatabase("a"), b"ns"),
            NamespacedDatabase(DirectoryBasedExampleDatabase("a"), b"ns"),
        ),
        (
            GitHubArtifactDatabase("owner1", "repo1"),
            GitHubArtifactDatabase("owner1", "repo1"),
        ),
    ],
)
def test_database_equal(db1, db2):
    assert db1 == db2


@pytest.mark.parametrize(
    "db1, db2",
    [
        (InMemoryExampleDatabase(), InMemoryExampleDatabase()),
        (InMemoryExampleDatabase(), DirectoryBasedExampleDatabase("a")),
        (BackgroundWriteDatabase(InMemoryExampleDatabase()), InMemoryExampleDatabase()),
        (DirectoryBasedExampleDatabase("a"), DirectoryBasedExampleDatabase("b")),
        (
            ReadOnlyDatabase(DirectoryBasedExampleDatabase("a")),
            ReadOnlyDatabase(DirectoryBasedExampleDatabase("b")),
        ),
        (
            NamespacedDatabase(DirectoryBasedExampleDatabase("a"), b"ns1"),
            NamespacedDatabase(DirectoryBasedExampleDatabase("a"), b"ns2"),
        ),
        (
            NamespacedDatabase(DirectoryBasedExampleDatabase("a"), b"ns"),
            NamespacedDatabase(DirectoryBasedExampleDatabase("b"), b"ns"),
        ),
        (
            GitHubArtifactDatabase("owner1", "repo1"),
            GitHubArtifactDatabase("owner2", "repo2"),
        ),
    ],
)
def test_database_not_equal(db1, db2):
    assert db1 != db2


@skipif_threading  # race in tmp_path
def test_directory_db_removes_empty_dirs(tmp_path):
    db = DirectoryBasedExampleDatabase(tmp_path)
    db.save(b"k1", b"v1")
    db.save(b"k1", b"v2")
    assert db._key_path(b"k1").exists()
    assert set(db.fetch(db._metakeys_name)) == {b"k1"}

    db.delete(b"k1", b"v1")
    assert db._key_path(b"k1").exists()
    assert set(db.fetch(db._metakeys_name)) == {b"k1"}

    db.delete(b"k1", b"v2")
    assert not db._key_path(b"k1").exists()
    assert set(db.fetch(db._metakeys_name)) == set()


@pytest.mark.parametrize("namespace", [b"", "", None, 0, bytearray(b"ns")])
def test_namespaced_database_requires_nonempty_bytes_namespace(namespace):
    with pytest.raises(InvalidArgument):
        NamespacedDatabase(InMemoryExampleDatabase(), namespace)


@given(namespace=st.binary(min_size=1), key=st.binary())
def test_encode_key_roundtrips(namespace, key):
    encoded = _encode_key(namespace, key)
    assert _decode_key(namespace, encoded) == key


@given(
    namespace1=st.binary(min_size=1),
    namespace2=st.binary(min_size=1),
    key=st.binary(),
)
def test_encode_key_isolated_between_namespaces(namespace1, namespace2, key):
    encoded = _encode_key(namespace1, key)
    if namespace1 == namespace2:
        assert _decode_key(namespace2, encoded) == key
    else:
        assert _decode_key(namespace2, encoded) is None


@given(encoded=st.binary())
def test_decode_key_never_raises(encoded):
    # the underlying database may contain arbitrary keys we did not write
    decoded = _decode_key(b"namespace", encoded)
    assert decoded is None or isinstance(decoded, bytes)


def test_namespaced_database_isolates_save_fetch_delete_move():
    shared = InMemoryExampleDatabase()
    db1 = NamespacedDatabase(shared, b"env-1")
    db2 = NamespacedDatabase(shared, b"env-2")

    db1.save(b"key", b"value1")
    db2.save(b"key", b"value2")

    assert set(db1.fetch(b"key")) == {b"value1"}
    assert set(db2.fetch(b"key")) == {b"value2"}
    # the underlying database has no entry under the unwrapped key
    assert set(shared.fetch(b"key")) == set()

    db1.delete(b"key", b"value1")
    assert set(db1.fetch(b"key")) == set()
    assert set(db2.fetch(b"key")) == {b"value2"}

    db2.move(b"key", b"other", b"value2")
    assert set(db2.fetch(b"key")) == set()
    assert set(db1.fetch(b"other")) == set()
    assert set(db2.fetch(b"other")) == {b"value2"}


@pytest.mark.parametrize(
    "namespaces",
    [
        # prefixes of each other
        [b"a", b"aa", b"aaa"],
        # null bytes and prefixes thereof
        [b"a", b"a\x00", b"a\x00b", b"\x00a", b"\x00", b"\x00\x00"],
        # bytes which look like our uleb128 length prefixes
        [b"\x7f", b"\x7f\x80", b"\x80", b"\x01", b"\x01\x01"],
        [b"x" * 127, b"x" * 128, b"x" * 129],
    ],
)
@pytest.mark.parametrize("key", [b"", b"\x00", b"key", b"\x01\x80\xff"])
def test_similar_namespaces_cannot_collide(namespaces, key):
    shared = InMemoryExampleDatabase()
    dbs = [NamespacedDatabase(shared, ns) for ns in namespaces]

    for i, db in enumerate(dbs):
        db.save(key, f"value-{i}".encode())
        # every earlier namespace still reads exactly its own value
        for j, other in enumerate(dbs[: i + 1]):
            assert set(other.fetch(key)) == {f"value-{j}".encode()}

    # each namespace maps to a distinct key in the underlying database
    underlying_keys = {_encode_key(ns, key) for ns in namespaces}
    assert len(underlying_keys) == len(namespaces)
    assert set(shared.data) == underlying_keys

    # deleting or moving one namespace's value leaves the others untouched
    dbs[0].delete(key, b"value-0")
    assert set(dbs[0].fetch(key)) == set()
    for i, db in enumerate(dbs[1:], start=1):
        assert set(db.fetch(key)) == {f"value-{i}".encode()}

    dbs[-1].move(key, b"dest", f"value-{len(dbs) - 1}".encode())
    for i, db in enumerate(dbs[1:-1], start=1):
        assert set(db.fetch(key)) == {f"value-{i}".encode()}
    assert set(dbs[-1].fetch(b"dest")) == {f"value-{len(dbs) - 1}".encode()}


@given(st.lists(st.tuples(st.binary(min_size=1), st.binary(), st.binary())))
@settings(max_examples=50)
def test_namespaced_database_matches_independent_databases(entries):
    # An arbitrary sequence of operations over N namespaces on one shared
    # database must give the same results as N independent databases.
    shared = InMemoryExampleDatabase()
    namespaced: dict[bytes, NamespacedDatabase] = {}
    reference: dict[bytes, dict[bytes, set[bytes]]] = {}

    for namespace, key, value in entries:
        if namespace not in namespaced:
            namespaced[namespace] = NamespacedDatabase(shared, namespace)
            reference[namespace] = {}
        namespaced[namespace].save(key, value)
        reference[namespace].setdefault(key, set()).add(value)

    for namespace, keys in reference.items():
        db = namespaced[namespace]
        for key, values in keys.items():
            assert set(db.fetch(key)) == values
            # direct access to the underlying database cannot see this key
            assert list(shared.fetch(key)) == []


@skipif_threading  # two DirectoryBasedExampleDatabase instances share a path
def test_namespaced_directory_databases_share_a_directory(tmp_path):
    db1 = NamespacedDatabase(DirectoryBasedExampleDatabase(tmp_path), b"env-a")
    db2 = NamespacedDatabase(DirectoryBasedExampleDatabase(tmp_path), b"env-b")

    db1.save(b"key", b"a")
    db2.save(b"key", b"b")

    assert set(db1.fetch(b"key")) == {b"a"}
    assert set(db2.fetch(b"key")) == {b"b"}

    db1.delete(b"key", b"a")
    assert set(db1.fetch(b"key")) == set()
    assert set(db2.fetch(b"key")) == {b"b"}


def test_namespaced_database_listener_only_sees_own_namespace():
    shared = InMemoryExampleDatabase()
    db1 = NamespacedDatabase(shared, b"ns1")
    db2 = NamespacedDatabase(shared, b"ns2")

    events: list = []
    db1.add_listener(events.append)

    db1.save(b"k", b"v1")
    db2.save(b"k", b"v2")
    db1.delete(b"k", b"v1")
    db2.delete(b"k", b"v2")
    db1.move(b"a", b"b", b"m")

    assert events == [
        ("save", (b"k", b"v1")),
        ("delete", (b"k", b"v1")),
        ("save", (b"b", b"m")),
    ]

    # deletes with unknown values are forwarded with the decoded original key
    encoded = _encode_key(b"ns1", b"k")
    shared._broadcast_change(("delete", (encoded, None)))
    assert events[-1] == ("delete", (b"k", None))

    # malformed or foreign keys broadcast directly on the underlying database
    # are filtered out and never cause errors
    namespace_prefix = _pack_uleb128(len(b"ns1")) + b"ns1"
    foreign_encoded = _encode_key(b"ns2", b"k")
    malformed_keys = [
        b"",
        b"\x80",
        b"\x03ns1",
        namespace_prefix,
        namespace_prefix + _pack_uleb128(1) + b"k" + b"junk",
        _pack_uleb128(4) + b"ns1\x00" + _pack_uleb128(0),
        foreign_encoded,
    ]
    before = list(events)
    for bad_key in malformed_keys:
        shared._broadcast_change(("save", (bad_key, b"v")))
        shared._broadcast_change(("delete", (bad_key, None)))
    assert events == before

    # removing the listener stops forwarding and unsubscribes from the
    # underlying database
    db1.remove_listener(events.append)
    assert shared._listeners == []
    db1.save(b"z", b"z")
    assert events == before


def test_namespaced_readonly_database():
    shared = InMemoryExampleDatabase()
    namespaced = NamespacedDatabase(ReadOnlyDatabase(shared), b"ns")
    namespaced.save(b"k", b"v")
    namespaced.move(b"k", b"k2", b"v")
    namespaced.delete(b"k", b"v")
    assert set(namespaced.fetch(b"k")) == set()
    assert shared.data == {}

    # change listening is still supported (it just never fires)
    def listener(event):
        raise AssertionError("ReadOnlyDatabase never fires change events")

    namespaced.add_listener(listener)
    namespaced.save(b"k2", b"v2")

    # and wrapping a NamespacedDatabase in ReadOnlyDatabase composes too
    wrapped = ReadOnlyDatabase(NamespacedDatabase(shared, b"ns2"))
    wrapped.save(b"k", b"v")
    assert set(wrapped.fetch(b"k")) == set()


def test_namespaced_multiplexed_database():
    a = InMemoryExampleDatabase()
    b = InMemoryExampleDatabase()
    multiplexed = MultiplexedDatabase(a, b)
    db1 = NamespacedDatabase(multiplexed, b"ns1")
    db2 = NamespacedDatabase(multiplexed, b"ns2")

    db1.save(b"k", b"v")
    db2.save(b"k", b"w")
    assert set(db1.fetch(b"k")) == {b"v"}
    assert set(db2.fetch(b"k")) == {b"w"}

    db1.move(b"k", b"k2", b"v")
    assert set(db1.fetch(b"k")) == set()
    assert set(db1.fetch(b"k2")) == {b"v"}
    assert set(db2.fetch(b"k")) == {b"w"}
    db1.delete(b"k2", b"v")
    assert set(db1.fetch(b"k2")) == set()
    assert set(db2.fetch(b"k")) == {b"w"}

    # listeners on one namespace only see that namespace's events (once per
    # backing database), reported with the caller's original key and value
    events: list = []
    db1.add_listener(events.append)
    db1.save(b"k3", b"v3")
    db2.save(b"k3", b"w3")
    assert Counter(events) == Counter([("save", (b"k3", b"v3"))] * 2)


def test_namespaced_background_write_database():
    backing = BackgroundWriteDatabase(InMemoryExampleDatabase())
    db1 = NamespacedDatabase(backing, b"ns1")
    db2 = NamespacedDatabase(backing, b"ns2")

    db1.save(b"k", b"v")
    db2.save(b"k", b"w")
    backing._join()
    assert set(db1.fetch(b"k")) == {b"v"}
    assert set(db2.fetch(b"k")) == {b"w"}

    events: list = []
    db1.add_listener(events.append)
    db1.save(b"k2", b"v2")
    db2.save(b"k2", b"w2")
    db1.move(b"k", b"moved", b"v")
    backing._join()

    assert ("save", (b"k2", b"v2")) in events
    assert not any(key == b"k2" and value == b"w2" for _, (key, value) in events)
    assert set(db1.fetch(b"moved")) == {b"v"}
    assert set(db2.fetch(b"k")) == {b"w"}

    db1.remove_listener(events.append)
    assert backing._db._listeners == []


def test_database_listener_namespaced_memory():
    _database_conforms_to_listener_api(
        lambda path: NamespacedDatabase(InMemoryExampleDatabase(), b"namespace"),
        parent_settings=settings(max_examples=5, stateful_step_count=10),
    )


def test_database_listener_namespaced_background_write():
    _database_conforms_to_listener_api(
        lambda path: NamespacedDatabase(
            BackgroundWriteDatabase(InMemoryExampleDatabase()), b"namespace"
        ),
        flush=lambda db: db._db._join(),
        parent_settings=settings(max_examples=5, stateful_step_count=10),
    )


def test_namespaced_database_repr():
    db = NamespacedDatabase(InMemoryExampleDatabase(), b"ns")
    assert "NamespacedDatabase" in repr(db)
    assert repr(b"ns") in repr(db)

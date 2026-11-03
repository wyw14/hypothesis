RELEASE_TYPE: minor

This release adds :func:`~hypothesis.database.compare_databases`, a read-only
helper which compares the examples stored under a given set of keys in two
:class:`~hypothesis.database.ExampleDatabase` instances. It reports the
examples found only in the left database, only in the right database, and in
both, deduplicated and sorted by byte value; it only uses the public
``fetch()`` method, so it works with any database backend.

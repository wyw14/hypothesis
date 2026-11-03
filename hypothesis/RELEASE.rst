RELEASE_TYPE: minor

This release adds :class:`~hypothesis.database.NamespacedDatabase`, a wrapper
which isolates an example database under a non-empty ``bytes`` namespace.
Multiple test environments can now share the same underlying database without
being able to read or delete each other's saved examples, and the wrapper
composes with :class:`~hypothesis.database.ReadOnlyDatabase`,
:class:`~hypothesis.database.MultiplexedDatabase`, and
:class:`~hypothesis.database.BackgroundWriteDatabase`.

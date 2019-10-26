# How to run

* Install a recent stable version of Rust (incl. cargo). Recommended way is via https://rustup.rs/
* Tasks are in the examples folder. You can run one with e.g.
``cargo run --example task01``.
Just replace the task name with the correct filename for the current task.
* Documentation for the code can be generated automatically with ``cargo doc --open``. Hopefully by documenting that possibility I'll actually write documentation comments...
* Tests are executed with ``cargo test``. At the moment that is only the examples from the documentation, though.

# Folder layout
* Plots and other images for a task are generated in a img_{taskname} folder.
* I'll put the task output along with the generated images into the results folder as well. That way you can look a them without running the tasks.

# Implemented Tasks
* Task 1, Subtask 1: ``cargo run --example task01``
* Task 2, Subtask 4: ``cargo run --example task02_4``
* Task 2, Subtask 5: ``cargo run --example task02_5``
* Task 2, Subtask 6: ``cargo run --example task02_6``
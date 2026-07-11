# Contributing to HindsightTag

HindsightTag is a research preview of an unvalidated mechanism. The most useful
contributions are ones that **test it against reality**: new host-system adapters,
harder benchmark scenarios, and hyperparameter studies — including negative results.

## Setting up

```bash
git clone https://github.com/vivekdudhat/hindsighttag.git
cd hindsighttag
pip install -e ".[all,dev]"
pytest tests/ -q
```

## Adding a new host-system adapter

This is the highest-impact contribution. Any memory system that can answer four
questions can attach HindsightTag.

1. **Create** `src/hindsighttag/adapters/<yourhost>_adapter.py` and subclass
   `hindsighttag.interface.HostMemorySystem`.

2. **Implement the four accessors** (paper Section 4.8 interface contract):

   ```python
   class MyHost(HostMemorySystem):
       def get_salience(self, item) -> float: ...          # host's own Sal(.) in [0,1]
       def get_timestamp(self, item) -> float: ...          # write time tau_i
       def get_relatedness(self, a, b) -> float: ...        # Rel(.,.) in [0,1]; if you
                                                            # can't, document omega_assoc=0
       def get_consolidation_threshold(self) -> float: ...  # decay-untouched cutoff
   ```

3. **Implement the storage primitives** the plug-in and benchmark drive:
   `store`, `tick`, `consolidate`, `get_memory`, `list_memories`, `is_retrievable`.

4. **Export it** from `src/hindsighttag/adapters/__init__.py` and, if it should be
   benchmarked, add it to `make_host()` in `benchmark/run_hrb.py`.

5. **Add tests**: at minimum, that a low-salience precursor followed by a
   high-salience trigger inside `W_capture` is rescued (see
   `tests/test_eq3_5_rescue.py` for the pattern), and that no rescue happens
   without a capture event (non-interference).

6. **Run the benchmark** against your adapter and include the numbers in your PR —
   good or bad. Negative results are accepted and valued.

Keep heavy dependencies optional (guarded imports + an extra in `pyproject.toml`),
following the pattern in `adapters/mem0_adapter.py`.

### Adding a new benchmark dataset

If you add LongMemEval or another public dataset:

1. **Do not commit the raw data** — add it to `.gitignore` and provide a download script.
2. **Document the license** in README under "Data & third-party attribution".
3. **Add a citation** (BibTeX block) for the dataset authors.
4. **Check NC/commercial restrictions** — LoCoMo is CC BY-NC 4.0; some datasets prohibit commercial use.
5. **Publish metrics only** in `benchmark/results/` — avoid committing raw conversation text from licensed datasets.

### Other valuable contributions

- **Benchmark extensions**: new scenario categories, longer delays, adversarial
  triggers, LongMemEval as a second base stream.
- **Hyperparameter studies**: sweeps beyond Section 5.3, especially of `T_tag`
  vs. delay distribution.
- **Bug reports** with a minimal reproduction.

## Ground rules

- Real numbers only: never hand-edit results files; every figure must be
  regenerable from a logged run.
- Match the paper's equations; if you change mechanism behaviour, cite the section
  you're implementing or explicitly flag the deviation.
- `pytest tests/ -q` must pass; add tests for anything you add.

## License

By contributing you agree your contributions are licensed under the MIT License.

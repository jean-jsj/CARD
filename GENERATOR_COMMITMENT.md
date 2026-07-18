# Generator commitment

The benchmark's data-generating process (simulation code, calibration pipeline, and calibrated parameter values) is **withheld during the evaluation phase**: together with the eval-seed values it is exactly the information that would let anyone regenerate the hidden truth for the eval seeds. The construction is fully documented at the equation level in the paper's DGP appendix.

To make the withholding verifiable rather than a "trust us":

- At release we archived the exact frozen generator source (deterministic tar of the generator tree at the freeze commit) and publish its digest here:

  ```
  SHA-256: 36746b849cd1c147a1e23482e7fb668f14480d690ccda789f81b4b55f83e7de5
  freeze commit: 668dc6b  (head of merged PR #97; tree-identical to the
           private repo's main branch tip 9a9b757)
  archive: card-generator-668dc6b.tar.gz  (243 files: simulation, calibration,
           validation, config, tests, text-generation pipeline; excludes only
           third-party-licensed raw inputs, which are never distributable)
  ```

- **We commit to publishing the archive itself after the evaluation phase** (at latest, upon completion of the associated paper's review cycle). Anyone can then verify `sha256sum` of the released archive against the digest above, proving the released generator is byte-identical to the one that produced the released data.

Until then, the released artifacts are the panels, the product-text surface, the scoring truth for the dev seed, and this scoring stack.

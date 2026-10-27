"""自检：权重在不在、依赖全不全、接口能不能跑通。

    Three questions, in the order that saves time:
      1. which methods does this build expose?
      2. for each one, is it weights / dependencies / upstream repo that is missing?
      3. can it actually produce an embedding right now?

    Run with ``python verify_install.py``; exits non-zero if anything is not ready.
"""
import sys

EXPECTED = ['deepprofiler', 'ca_mae', 'cellclip', 'o2vae']


def payload_for(models, name):
    """Build a legal payload for ``name`` from its own manifest.

    Three input conventions coexist and none is guessable from a method name:
    pixel crops (layout and channel count vary per method), precomputed channel
    embeddings, and nucleus silhouettes.  One method also refuses raw crops
    outright because its centring happens upstream in MATLAB.  Reading the
    manifest is the point -- a caller has to do the same, so this self-check
    demonstrates the intended path instead of hiding it.
    """
    import numpy as np

    m = models.manifest(name)
    n_channels = len(m.channels.names)
    height, width = m.spatial or (256, 256)

    if m.input_unit.value == "channel_embeddings":
        return np.full((2, n_channels, m.input_feature_dim or 1536), 0.5,
                       dtype="float32"), None, None, None

    if m.preprocess.source_map == "matlab_centring":
        # prepare() refuses by design: the upstream pipeline centres these crops.
        # Supply them already centred and already in the declared value range.
        return (np.full((2, n_channels, height, width), 0.5, dtype="float32"),
                None, None, None)

    # A constant mid-range value, not zeros: per-crop normalisation divides by
    # the crop maximum, so an all-zero crop would turn a readiness report into
    # a NaN report.
    raw = np.full((2, height, width, n_channels), 1000, dtype="uint16")
    channels = tuple(m.channels.names) if m.channels.channel_count_fixed else None
    masks = None
    if m.preprocess.mask_mode != "none" or m.preprocess.source_map == "mask_to_silhouette":
        masks = np.ones((2, height, width), dtype="uint8")
    return raw, "raw_crop", channels, masks


def main() -> int:
    from cellpaint_pipeline import models

    print("profile :", models.active_profile())
    declared = models.declared()
    print("methods :", ", ".join(declared))

    missing = [m for m in EXPECTED if m not in declared]
    if missing:
        print("!! methods missing from this build:", missing)
        return 1

    print("\n--- readiness ---")
    not_ready = 0
    for row in models.doctor():
        ok = row["status"] == "ready"
        not_ready += 0 if ok else 1
        print("  %-13s %-22s %s" % (row["model"], row["status"],
                                    row["weights_detail"] or "-"))
    if not_ready:
        print("\n%d method(s) not ready. Put the weights in place (README section 2)" % not_ready)
        print("or install the missing framework (README section 1).")
        return 1

    print("\n--- real forward pass ---")
    failures = 0
    for name in declared:
        try:
            payload, assume, channels, masks = payload_for(models, name)
            out = models.embed(name, payload, assume=assume, channels=channels,
                               masks=masks, device="cpu")
            print("  %-13s -> %-12s provenance=%s" % (
                name, out.as_numpy().shape, out.metadata.get("weights_provenance")))
        except Exception as exc:  # noqa: BLE001 - a self-check reports, never raises
            failures += 1
            print("  %-13s !! %s: %s" % (name, type(exc).__name__, str(exc)[:70]))
    if failures:
        print("\n%d method(s) loaded but could not run. See the message above." % failures)
        return 1
    print("\nall good.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

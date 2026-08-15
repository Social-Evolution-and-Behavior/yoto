# Tuning API

The functions behind `yoto train`. See the
[Training and Tuning guide](../guides/training.md) for the workflow.

::: yoto.tuning.testset
    options:
      show_source: true
      members:
        - build_testset

::: yoto.tuning.optimize
    options:
      show_source: true
      members:
        - optimize_preset
        - optimize_preset_images
        - subsample_testset

::: yoto.tuning.crop_dataset
    options:
      show_source: true
      members:
        - build_crop_dataset

::: yoto.tuning.viz
    options:
      show_source: true
      members:
        - render_comparison

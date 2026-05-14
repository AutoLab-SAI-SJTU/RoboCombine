# Third-Party Notices

RoboCombine contains original evaluation glue code plus task
environment files and policy-adapter interfaces derived from, or designed to run
with, RoboTwin-compatible simulation assets and external policy runtimes.

## Components

| Component | Role | Included in this package | Notice |
| --- | --- | --- | --- |
| RoboTwin-compatible simulator/task API | Provides the robot task runtime expected by `envs/` and `script/eval_policy.py`. | Partially, as task/runtime wrappers. Full simulator assets are not included. | Retains its upstream license and attribution. |
| Object, embodiment, texture, and scene assets | Required to render and execute the tasks. | Not included. | Users must obtain these assets from their original permitted source and comply with their licenses. |
| Motus policy runtime assets | Optional policy adapter target. | Adapter code is included. Runtime env, checkpoints, WAN assets, and VLM assets follow upstream Motus requirements and are not bundled here. | External assets retain their original licenses. |
| pi05/OpenPI policy runtime assets | Optional policy adapter target. | Adapter code only. Checkpoints are not included. | External assets retain their original licenses. |
| X-VLA progress predictor | Optional real-switch predictor backend. | Runtime interface only. Predictor bundle and backbone checkpoint are not included. | External assets retain their original licenses. 
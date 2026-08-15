import collections
import dataclasses
import logging
import math
import pathlib
import time
import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.zmq_gr00t import RobotInferenceClient
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8001
    resize_size: int = 224
    replan_steps: int = 8

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_10"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )# libero_10
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 10  # Number of rollouts per task
    real_time_chunking = False
    #################################################################################################################
    # Utils
    #################################################################################################################
    # video_out_path: str = "data/libero/videos_vita_spatial"  # Path to save videos
    video_out_path: str = "data/vita/finetuned_vita_libero_pretrained_finetuned_mixed_bs64_gpu8/libero_10"  # Path to save videos
    # video_out_path: str = "data/libero/videos_gr00t_n1.5"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    # client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    client = RobotInferenceClient(host=args.host, port=args.port)
    # print("Available modality config available:")
    # modality_configs = client.get_modality_config()
    # print(modality_configs.keys())

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            
            if args.real_time_chunking:
                prev_action_chunk = None

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    # img = image_tools.convert_to_uint8(
                    #     image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    # )
                    # wrist_img = image_tools.convert_to_uint8(
                    #     image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    # )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        if args.real_time_chunking:
                            element = {
                                "video.image": np.expand_dims(img, axis=0),
                                "video.wrist_image": np.expand_dims(wrist_img, axis=0),
                                "state.state": np.expand_dims(
                                    np.concatenate(
                                        (
                                            obs["robot0_eef_pos"],
                                            _quat2axisangle(obs["robot0_eef_quat"]),
                                            obs["robot0_gripper_qpos"],
                                        )
                                    ),
                                    axis=0
                                ),
                                "annotation.human.task_description": [str(task_description)],
                                "prev_action_chunk": prev_action_chunk,
                            }
                            
                            realtime_action = client.get_realtime_action(element)
                            action_chunk=realtime_action["action.actions"]
                            prev_action_chunk=realtime_action["prev_action_chunk"]
                        else:
                            # Finished executing previous action chunk -- compute new chunk
                            # Prepare observations dict
                            element = {
                                "video.image": np.expand_dims(img, axis=0),
                                "video.wrist_image": np.expand_dims(wrist_img, axis=0),
                                "state.state": np.expand_dims(
                                    np.concatenate(
                                        (
                                            obs["robot0_eef_pos"],
                                            _quat2axisangle(obs["robot0_eef_quat"]),
                                            obs["robot0_gripper_qpos"],
                                        )
                                    ),
                                    axis=0
                                ),
                                "annotation.human.task_description": [str(task_description)],
                            }
                            # Query model to get action
                            # start_time = time.time()
                            action_chunk = client.get_action(element)["action.actions"]
                            
                        # end_time = time.time()
                        # execution_time_ms = (end_time - start_time) * 1000
                        # print(f"client.get_action 的总执行时间: {execution_time_ms:.2f} ms", flush=True)
                        # if args.task_suite_name != "libero_90":
                        action_chunk[:,-1] = -2*action_chunk[:,-1] + 1
                        
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])
                        
                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")

def unchunk(action_chunk, action_plan, replan_steps):
    lengths = [v.shape[0] for v in action_chunk.values()]
    action_chunk['action.gripper'] = -2*action_chunk['action.gripper'] + 1
    if len(set(lengths)) != 1:
        raise ValueError(f"Inconsistent lengths: {lengths}")
    assert (
        lengths[0] >= replan_steps
    ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
    # T = lengths[0]
    T = replan_steps

    keys = list(action_chunk.keys())
    for t in range(T):
        # Take the scalar at index t from each array, and put it into an ndarray of (D,) on axis=0.
        new_arr = np.stack([action_chunk[k][t] for k in keys], axis=0)
        # new_arr[-1] = -new_arr[-1]
        action_plan.append(new_arr)
    return action_plan

def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)

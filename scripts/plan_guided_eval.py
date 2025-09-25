import pdb
import numpy as np
import diffuser.sampling as sampling
import diffuser.utils as utils
from scipy.stats import gaussian_kde

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'walker2d-medium-replay-v2'
    config: str = 'config.locomotion'

args = Parser().parse_args('plan')

#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#

## load diffusion model and value function from disk
diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed,
)
value_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.value_loadpath,
    epoch=args.value_epoch, seed=args.seed,
)

utils.check_compatibility(diffusion_experiment, value_experiment)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

value_function = value_experiment.ema
guide_config = utils.Config(args.guide, model=value_function, verbose=False)
guide = guide_config()

logger_config = utils.Config(
    utils.Logger,
    renderer=renderer,
    logpath=args.savepath,
    vis_freq=args.vis_freq,
    max_render=args.max_render,
)

policy_config = utils.Config(
    args.policy,
    guide=guide,
    scale=args.scale,
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
    sample_fn=sampling.n_step_guided_p_sample,
    n_guide_steps=args.n_guide_steps,
    t_stopgrad=args.t_stopgrad,
    scale_grad_by_std=args.scale_grad_by_std,
    verbose=False,
)

logger = logger_config()
policy = policy_config()

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

env = dataset.env
observation = env.reset()

rollout = [observation.copy()]
all_errors = []

predicted_trajectories = []

max_horizon = diffusion.horizon

for t in range(300):

    if t % 10 == 0: print(args.savepath, flush=True)

    state = env.state_vector().copy()
    conditions = {0: observation}
    action, samples = policy(conditions, batch_size=args.batch_size, verbose=args.verbose)

    predicted_trajectories.append(samples.observations)  # shape: [1, horizon, obs_dim]

    next_observation, reward, terminal, _ = env.step(action)

    rollout.append(next_observation.copy())

    ## compute backtrace prediction errors based on current timestep
    backtrace_errors = {}
    for b in range(1, min(len(predicted_trajectories)+1, max_horizon+1)):
        t_prev = t - b
        if t_prev < 0: continue
        pred_traj = predicted_trajectories[t_prev][0]  # shape: [horizon, obs_dim]
        if pred_traj.shape[0] <= b-1: continue
        pred = pred_traj[b-1]  # the prediction made b steps ago about now
        real = observation
        error = np.linalg.norm(pred - real)
        backtrace_errors[b] = error

    if len(backtrace_errors) > 0:
        print(f"[Prediction Errors @ t={t}] " + " | ".join([
            f"{k}-step: {v:.3f}" for k, v in sorted(backtrace_errors.items())]), flush=True)

    all_errors.append(backtrace_errors)
    logger.log(t, samples, state, rollout)

    if terminal:
        break

    observation = next_observation

logger.finish(t, reward, reward, terminal, diffusion_experiment, value_experiment)

#-----------------------------------------------------------------------------#
#---------------------------- post-run analysis ------------------------------#
#-----------------------------------------------------------------------------#

if len(all_errors) >= 64:
    all_keys = sorted(set(k for step in all_errors[-64:] for k in step.keys()))
    quarter_means = []
    for i in range(4):
        q_errors = []
        for step in all_errors[-64 + i*16: -64 + (i+1)*16]:
            for k in all_keys:
                if k in step:
                    q_errors.append(step[k])
        quarter_means.append(np.mean(q_errors))

    print("\n[Quarter-wise Average Prediction Errors over last 64 steps]")
    for i, avg in enumerate(quarter_means):
        print(f"Quarter {i+1} ({i*16+1}–{(i+1)*16}): {avg:.3f}")


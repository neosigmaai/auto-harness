"""
Per-task WebArena driver. Infrastructure — not optimized by the coding agent.

`WebArenaRunner` subprocess-invokes this script once per task. Each invocation:
  1. Loads the cloned WebArena source (adds it to sys.path, then chdir into it
     so WebArena's relative paths — JSON prompts, auth cookies, config files —
     all resolve correctly).
  2. Instantiates `agent.agent.HarnessAgent` (subclass of WebArena's PromptAgent).
  3. Resets a Playwright ScriptBrowserEnv with the task's config_file.
  4. Drives the action loop with early-stop guards matching WebArena's run.py.
  5. Calls evaluation_harness.evaluator_router to produce a binary reward.
  6. Writes <out-dir>/result.json always, <out-dir>/trace.json iff --save-trace=1.

Never prints observations or config details to stdout: the parent runner forwards
stdout, and test observations must not leak into the coding agent's view.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WebArena per-task driver")
    p.add_argument("--task-id", required=True)
    p.add_argument("--config-file", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--save-trace", type=int, choices=[0, 1], default=0)
    p.add_argument("--webarena-repo", default="webarena_repo")
    p.add_argument("--model", default=os.getenv("AGENT_MODEL", "gpt-4o-mini"))
    p.add_argument("--action-set-tag", default="id_accessibility_tree")
    p.add_argument("--observation-type", default="accessibility_tree")
    return p.parse_args()


def _write_result(out_dir: str, payload: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(payload, f, indent=2)


def _write_trace(out_dir: str, messages: list) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "trace.json"), "w") as f:
        json.dump(messages, f, indent=2, default=str)


def main() -> int:
    args = _parse_args()
    # Driver lives at `<repo_root>/webarena/driver.py`, so repo_root is the
    # parent of this file's directory. It's used below to locate
    # `<repo_root>/agent/agent.py` (the coding agent's HarnessAgent).
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    webarena_dir = os.path.abspath(args.webarena_repo)
    config_file = os.path.abspath(args.config_file)
    out_dir = os.path.abspath(args.out_dir)
    save_trace = bool(args.save_trace)

    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    # Preflight — fail loudly before we touch Playwright.
    if not os.path.isdir(webarena_dir):
        _write_result(out_dir, {
            "task_id": args.task_id,
            "reward": None,
            "error": f"webarena repo not found: {webarena_dir}",
            "duration_sec": 0.0,
        })
        return 2
    if not os.path.exists(config_file):
        _write_result(out_dir, {
            "task_id": args.task_id,
            "reward": None,
            "error": f"config_file not found: {config_file}",
            "duration_sec": 0.0,
        })
        return 2

    # WebArena uses relative paths for its prompt JSONs and auth cookies, and
    # its import layout (agent/, browser_env/, evaluation_harness/) lives at
    # repo root. Enter the clone directory for the duration of the run.
    #
    # Import-collision hazard: both repo_root/agent/ and webarena_dir/agent/
    # define module `agent.agent`. The coding-agent file at
    # repo_root/agent/agent.py must resolve to OUR HarnessAgent, but its own
    # `from agent.agent import PromptAgent` line (and WebArena's subpackages
    # like agent.prompts.*, browser_env.*, evaluation_harness.*) must resolve
    # to WebArena's clone. We can't have both live under the same `agent.agent`
    # module name.
    #
    # Resolution: put webarena_dir on sys.path so `agent.agent` resolves to
    # WebArena's PromptAgent — then load our HarnessAgent class directly from
    # its file path via importlib, bypassing the module-name lookup entirely.
    old_cwd = os.getcwd()
    sys.path = [p for p in sys.path if p not in ("", ".")]
    sys.path.insert(0, webarena_dir)
    os.chdir(webarena_dir)

    reward: float | None = None
    error: str | None = None
    n_steps = 0
    action_history: list[str] = []
    env = None  # set inside try; released in finally

    try:
        # Imports are inside try so missing webarena deps surface as infra errors.
        from browser_env import (  # noqa: E402
            ActionTypes,
            ScriptBrowserEnv,
            StateInfo,
            Trajectory,
            create_stop_action,
        )
        from browser_env.actions import is_equivalent  # noqa: E402
        from browser_env.helper_functions import get_action_description  # noqa: E402
        from evaluation_harness import evaluator_router  # noqa: E402

        # Load the coding-agent's HarnessAgent by file path so it doesn't
        # collide with WebArena's own `agent.agent` module (see note above).
        import importlib.util
        harness_path = os.path.join(repo_root, "agent", "agent.py")
        spec = importlib.util.spec_from_file_location(
            "harness_agent_module", harness_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load HarnessAgent from {harness_path}")
        _harness_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_harness_mod)
        HarnessAgent = _harness_mod.HarnessAgent

        with open(config_file) as f:
            cfg = json.load(f)
        intent = cfg["intent"]

        env = ScriptBrowserEnv(
            headless=True,
            slow_mo=0,
            observation_type=args.observation_type,
            current_viewport_only=True,
            viewport_size={"width": 1280, "height": 720},
            save_trace_enabled=False,
            sleep_after_execution=0.0,
        )

        agent = HarnessAgent(
            model=args.model,
            action_set_tag=args.action_set_tag,
        )
        agent.reset(config_file)

        trajectory: Trajectory = []
        obs, info = env.reset(options={"config_file": config_file})
        state_info: StateInfo = {"observation": obs, "info": info}
        trajectory.append(state_info)

        meta_data = {"action_history": ["None"]}
        max_steps = args.max_steps

        parsing_failure_th = 3
        repeating_action_th = 3

        def _early_stop(traj):
            steps = (len(traj) - 1) / 2
            if steps >= max_steps:
                return True, f"Reach max steps {max_steps}"
            last_k = traj[1::2][-parsing_failure_th:]
            if len(last_k) >= parsing_failure_th and all(
                a["action_type"] == ActionTypes.NONE for a in last_k
            ):
                return True, f"Failed to parse actions for {parsing_failure_th} times"
            action_seq = traj[1::2]
            if not action_seq:
                return False, ""
            last_action = action_seq[-1]
            if last_action["action_type"] != ActionTypes.TYPE:
                last_k = action_seq[-repeating_action_th:]
                if len(last_k) >= repeating_action_th and all(
                    is_equivalent(a, last_action) for a in last_k
                ):
                    return True, f"Same action for {repeating_action_th} times"
            else:
                if sum(
                    is_equivalent(a, last_action) for a in action_seq
                ) >= repeating_action_th:
                    return True, f"Same typing action for {repeating_action_th} times"
            return False, ""

        while True:
            stop_flag, stop_info = _early_stop(trajectory)
            if stop_flag:
                action = create_stop_action(f"Early stop: {stop_info}")
            else:
                try:
                    action = agent.next_action(
                        trajectory, intent, meta_data=meta_data
                    )
                except ValueError as e:
                    action = create_stop_action(f"ERROR: {e}")

            trajectory.append(action)
            n_steps += 1

            action_str = get_action_description(
                action,
                state_info["info"]["observation_metadata"],
                action_set_tag=args.action_set_tag,
                prompt_constructor=getattr(agent, "prompt_constructor", None),
            )
            meta_data["action_history"].append(action_str)
            action_history.append(action_str)

            if action["action_type"] == ActionTypes.STOP:
                break

            obs, _, terminated, _, info = env.step(action)
            state_info = {"observation": obs, "info": info}
            trajectory.append(state_info)

            if terminated:
                trajectory.append(create_stop_action(""))
                break

        evaluator = evaluator_router(config_file)
        score = evaluator(
            trajectory=trajectory,
            config_file=config_file,
            page=env.page,
            client=env.get_page_client(env.page),
        )
        reward = float(score)

        # Persist trace only when explicitly allowed (train split).
        if save_trace:
            # Strip heavy observation blobs; the coding agent needs actions +
            # short info, not the full accessibility tree per step.
            trace_events = []
            for i, event in enumerate(trajectory):
                if isinstance(event, dict) and "action_type" in event:
                    trace_events.append({
                        "step": i,
                        "kind": "action",
                        "action_type": int(event.get("action_type", 0)),
                        "raw_prediction": event.get("raw_prediction", ""),
                    })
                elif isinstance(event, dict) and "observation" in event:
                    obs_any = event["observation"]
                    # observation is a dict when observation_type=accessibility_tree
                    text = (obs_any.get("text", "") if isinstance(obs_any, dict)
                            else str(obs_any))
                    trace_events.append({
                        "step": i,
                        "kind": "observation",
                        "text_preview": text[:2000],
                    })
            _write_trace(out_dir, {
                "task_id": args.task_id,
                "intent": intent,
                "action_history": action_history,
                "events": trace_events,
            })

    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        # Log full traceback to out_dir for diagnostics; the parent runner only
        # surfaces a summary line so the coding agent doesn't see test-split
        # internals.
        with open(os.path.join(out_dir, "error.txt"), "w") as f:
            f.write(f"{error}\n\n{traceback.format_exc()}")
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        os.chdir(old_cwd)

    _write_result(out_dir, {
        "task_id": args.task_id,
        "reward": reward,
        "steps": n_steps,
        "error": error,
        "duration_sec": round(time.time() - t0, 2),
    })

    # Exit 0 even on task failure — only exit non-zero on infra errors so the
    # parent runner can distinguish.
    return 0 if error is None else 1


if __name__ == "__main__":
    sys.exit(main())

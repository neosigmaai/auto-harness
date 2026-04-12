Main challenge is that SWE-Bench is far more complex than Tau Bench. It is more expensive to run, takes longer, uses Docker, nad has its own harness for an agent. This brings up two main concerns.

1. How do we faithfully have agent.py still be autonomous + become a harness for SWE Bench?
2. How do we get quicker evaluations and results.

1st: Let's limit the # of tasks to only 30 coding tasks from SWE Bench Lite. 

We want to keep gating logic the same which is fine with SWEBench which will output score of 1 or 0. 

We plan on now using mini-swe-agent as the inner code loop and having swe_agent.py be a wrapper for it where we can alter certain modules. This gives it certain loops and 'architectures' to use.

It works much better! One thing we know want to do is to know how well we expect mini-SWE-agent to perform without being a wrapper to see if our implementation is ineffective or wrong.

At 8:27, we have 4/23 right for our base model which uses mini-swe-agent as a wrapper. The mini-swe-agent by itself gets 6/23 right so we are not performing at baseline.

8:40, looked at differences between the two: Increased step count for our implementation as well docker inside of docker so the enviornment can be recreated faithfully.

8:50, refactoring + kept information in PROGRAM.md split into SWE-PROGRAM.md and TAU-PROGRAM.md

8:50, Rerunning evaluation on our agent.

8:58, evaluation still running. Modifying swe_runner.py so that we pass more context to our agent directly, specifically we inject that they must make this pass: test/cli/commands_test.py::test__cli__command_directed and hint.

9:17, it finished evaluating. We get 5/23 correct.
9:20, for quicker iteration, we create a mini:true flag inside of our experiment_config.yaml file.

On the three, we get 2/3 right.

9:30, Claude Code is running to try and make swe_agent.py more effective (new prompting).

9:40, Claude analyzes: ⏺ Now I see the exact issue: the test compares the L031 rule message text. Expected: "Avoid aliases in from clauses and join conditions." but actual: "Avoid using aliases in join condition". The agent fixed the wrong thing (colorize instead of
  the L031 rule message).


Couple key design choices:

1. We started simple with just an LLM, prompt, and problem overview and a one-pass output. This proved highly ineffective. The mindset we have was to not-over-engineer a solution. I shared with Ritvik how from my experience, over-engineering is a huge problem. In this case, I wanted to avoid having our autonomous agent build an entire new replica of swe-agent.
2. We knew that time was a limit, so I only focused on testing from SWE-lite database, and only the dev portion of the database. Later on, we only tested 3 tests because it would save even more time.
3. We tested SWE-lite just to compare if our implemeentation was worse. We went from 3/23 to 5/23 by asking what was different from SWE-lite and our wrapper implementation.
4. Since the information from a SWE-bench test is quite dense and also not necessarily very helpful to an agent, we created summarization of runs in last_run_summary.md. 
5. Inevitably, we had to create a docker enviornment inside of a docker enviornment so our agent could render the right dependencies and files. Before, it created a git clone but had to install dependencies manually.
6. Claude seems to have a bias towards changing the prompt which might overfit the solution. I think in future iterations, I would seed with 'compare your implementation' to the full mini-SWE-agent more. AND think about tooling.
# Recording demo

Use an environment with the project's core runtime dependencies plus Streamlit. The demo preparation was tested on Python 3.12 with Streamlit 1.66, NumPy 2.3.5 and pandas 3.0.1; some full-project requirements differ. These versions describe the tested demo environment, not full training reproducibility.

Run from this repository:

```powershell
python -m streamlit run demo_app.py
```

Choose a simulated profile and click Compare baselines. This runs random and rule-based policies, not DQN/PPO. No clinical validation or medical use.

Record a 60–120 second walkthrough with Windows Snipping Tool (Win+Shift+R), show a real interaction and its output, then save and review the video. Explain the implementation was AI-assisted. Do not show secrets or private records. A working demo does not establish every feature or reported experiment works.

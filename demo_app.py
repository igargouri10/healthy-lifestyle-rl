"""Live baseline simulation. DQN/PPO training is a separate workflow."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
import pandas as pd
import streamlit as st
from lifestyle_rl.agents.baselines import RandomPolicy,RuleBasedPolicy
from lifestyle_rl.envs.healthy_lifestyle_env import HealthyLifestyleEnv
from lifestyle_rl.simulation.user_profiles import PROFILE_LIBRARY

st.title('Lifestyle RL: inspect the simulator')
st.warning('Academic teaching simulator. No clinical validation or medical advice. This demo compares non-learning baselines, not trained DQN/PPO.')
st.caption('AI-assisted implementation | Simulated profiles only')
profile=st.selectbox('Simulated profile',list(PROFILE_LIBRARY))
seed=st.number_input('Seed',0,10000,42)
if st.button('Compare baselines'):
    results=[]
    for name,policy in [('Random',RandomPolicy(seed=int(seed))),('Rule-based',RuleBasedPolicy())]:
        env=HealthyLifestyleEnv(profile_name=profile); obs,info=env.reset(seed=int(seed)); done=False; day=0
        while not done:
            obs,reward,term,trunc,info=env.step(policy.act(obs,info)); done=term or trunc; day+=1
            results.append({'policy':name,'day':day,'reward':reward,'intervention':int(info['safety_intervention_applied']),'recovery':info['recovery_score']})
    st.session_state['results']=pd.DataFrame(results)
if 'results' in st.session_state:
    df=st.session_state['results']
    st.dataframe(df.groupby('policy').agg(total_reward=('reward','sum'),safety_interventions=('intervention','sum')))
    st.line_chart(df.pivot(index='day',columns='policy',values='recovery'))
    st.dataframe(df,hide_index=True)
    st.download_button('Download simulation trace',df.to_csv(index=False),'baseline-trace.csv','text/csv')

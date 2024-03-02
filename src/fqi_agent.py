from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
import numpy as np
import tqdm
import joblib

class FQIAgent:
    def __init__(self, config) -> None:
        self.nb_actions = config['nb_actions']
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.98
        self.horizon = config['horizon'] if 'horizon' in config.keys() else int(1e4)
        self.disable_tqdm = config['disable_tqdm'] if 'disable_tqdm' in config.keys() else False
        self.regressor_name = config['regressor_name'] if 'regressor_name' in config.keys() else 'ExtraTreesRegressor'
        self.regressor_params = config['regressor_params']
        self.monitor_every = config['monitor_every'] if 'monitor_every' in config.keys() else 5
        self.monitoring_nb_trials = config['monitoring_nb_trials'] if 'monitoring_nb_trials' in config.keys() else 20

    def greedy_action(self, s):
        Qsa = []
        for a in range(self.nb_actions):
            sa = np.append(s,a).reshape(1, -1)
            Qsa.append(self.Q.predict(sa))
        return np.argmax(Qsa)
    
    def MC_eval(self, env, nb_trials):   # NEW NEW NEW
        MC_total_reward = []
        MC_discounted_reward = []
        for _ in range(nb_trials):
            x,_ = env.reset()
            done = False
            trunc = False
            total_reward = 0
            discounted_reward = 0
            step = 0
            while not (done or trunc):
                a = self.greedy_action(x)
                y,r,done,trunc,_ = env.step(a)
                x = y
                total_reward += r
                discounted_reward += self.gamma**step * r
                step += 1
            MC_total_reward.append(total_reward)
            MC_discounted_reward.append(discounted_reward)
        return np.mean(MC_discounted_reward), np.mean(MC_total_reward)

    def init_regressor(self):
        if self.regressor_name == 'ExtraTreesRegressor':
            return ExtraTreesRegressor(**self.regressor_params)
        elif self.regressor_name == 'RandomForestRegressor':
            return RandomForestRegressor(**self.regressor_params)

    def train(self, env, iterations):
        print(f"Collecting samples: ")
        S, A, R, S2, D = self.collect_samples(env)
        nb_samples = S.shape[0]
        SA = np.append(S, A, axis=1)
        MC_avg_total_reward = []   # NEW NEW NEW
        MC_avg_discounted_reward = []   # NEW NEW NEW
        print(f"Training: ")
        for iter in tqdm(range(iterations), disable=self.disable_tqdm):
            if iter==0:
                value=R.copy()
            else:
                Q2 = np.zeros((nb_samples, self.nb_actions))
                for a2 in range(self.nb_actions):
                    A2 = np.full_like(A, a2)
                    S2A2 = np.append(S2,A2,axis=1)
                    Q2[:,a2] = self.Q.predict(S2A2)
                max_Q2 = np.max(Q2,axis=1)
                value = R + self.gamma*(1-D)*max_Q2
            if iter > 0 and iter % self.monitor_every == 0:
                MC_dr, MC_tr = self.MC_eval(env, self.monitoring_nb_trials)    # NEW NEW NEW
                MC_avg_total_reward.append(MC_tr)
                MC_avg_discounted_reward.append(MC_dr)
            self.Q = self.init_regressor()
            self.Q.fit(SA, value)
        return MC_avg_discounted_reward, MC_avg_total_reward

    def collect_samples(self, env, print_done_states=False):
        s, _ = env.reset()
        S = []
        A = []
        R = []
        S2 = []
        D = []
        for _ in tqdm(range(self.horizon), disable=self.disable_tqdm):
            a = env.action_space.sample()
            s2, r, done, trunc, _ = env.step(a)
            S.append(s)
            A.append(a)
            R.append(r)
            S2.append(s2)
            D.append(done)
            if done or trunc:
                s, _ = env.reset()
                if done and print_done_states:
                    print("done!")
            else:
                s = s2
        S = np.array(S)
        A = np.array(A).reshape((-1,1))
        R = np.array(R)
        S2= np.array(S2)
        D = np.array(D)
        return S, A, R, S2, D
    

    def act(self, state):
        return self.greedy_action(state)
    
    def save(self, path):
        joblib.dump(self.Q, path)

    def load(self, path):
        self.Q = joblib.load(path)
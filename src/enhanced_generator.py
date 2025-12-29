"""
JMES Enhanced Synthetic Data Generator - Complete
==================================================
150,000 personnel, 10 years, full contingency modelling
"""

import numpy as np
import pandas as pd
import yaml
from typing import Dict, List, Tuple, Optional, Any
import uuid
from collections import defaultdict
import gc

from enhanced_models import (
    EnhancedPersonnelMaster, EnhancedPersonMonth, InjuryHistoryRecord,
    ServiceBranch, ServiceType, Gender, Ethnicity, RankBand, RankCategory,
    Trade, JMESStatus, InjuryType, BodyRegion, DeploymentType, TrainingPhase,
    PregnancyStatus, EngagementType, UnitEnvType, HealthRisk, OutflowReason,
    ChronicCondition
)


class EnhancedSyntheticGenerator:
    def __init__(self, config_path: str, seed: int = 42):
        self.config = self._load_config(config_path)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.active_personnel: Dict[str, Dict] = {}
        self.master_records: List[Dict] = []
        self.month_records: List[Dict] = []
        self.injury_records: List[Dict] = []
        self._build_lookups()
        self.stats = defaultdict(int)
    
    def _load_config(self, path: str) -> dict:
        with open(path) as f:
            return yaml.safe_load(f)
    
    def _build_lookups(self):
        cfg = self.config
        # Ethnicity
        eth = cfg.get('ethnicity', {}).get('distribution', {'White_British': 1.0})
        self.eth_opts = list(eth.keys())
        self.eth_probs = np.array(list(eth.values()))
        self.eth_probs /= self.eth_probs.sum()
        # Trades
        trades = {k: v for k, v in cfg.get('trades', {}).items() 
                  if isinstance(v, (int, float)) and k != 'risk_profiles'}
        self.trade_opts = list(trades.keys())
        self.trade_probs = np.array(list(trades.values()))
        self.trade_probs /= self.trade_probs.sum()
        # Body regions
        body = cfg.get('injuries', {}).get('body_regions', {'lower_back': 1.0})
        self.body_opts = list(body.keys())
        self.body_probs = np.array(list(body.values()))
        self.body_probs /= self.body_probs.sum()
        # Injury types
        inj = cfg.get('injuries', {}).get('types', {})
        self.inj_opts = list(inj.keys())
        self.inj_probs = np.array([inj[k].get('probability', 0.1) for k in self.inj_opts])
        self.inj_probs /= self.inj_probs.sum()
    
    def _choice(self, opts: List, probs: np.ndarray) -> Any:
        idx = self.rng.choice(len(opts), p=probs)
        return opts[idx]
    
    def _choice_dict(self, d: Dict[str, float]) -> str:
        keys = list(d.keys())
        probs = np.array(list(d.values()))
        probs /= probs.sum()
        return keys[self.rng.choice(len(keys), p=probs)]
    
    def generate_person(self, is_new: bool = False, cohort_month: int = 0) -> EnhancedPersonnelMaster:
        cfg = self.config
        
        # Service
        service = self._choice_dict(cfg['service_mix'])
        svc_prof = cfg.get('service_profiles', {}).get(service, {})
        
        # Gender
        fem_rate = cfg['gender']['overall_female_rate'] * svc_prof.get('female_rate_modifier', 1.0)
        gender = Gender.FEMALE if self.rng.random() < fem_rate else Gender.MALE
        
        # Trade (gender-weighted)
        trade = self._gen_trade(gender)
        trade_risk = cfg.get('trades', {}).get('risk_profiles', {}).get(trade, {})
        
        # Age/LoS
        if is_new:
            age_cfg = cfg['age']['inflow']
            age = int(np.clip(self.rng.normal(age_cfg['mean'], age_cfg['std']), 18, 35))
            los = min(self.rng.integers(0, 3), age - 18)  # Ensure LoS <= age - 18
        else:
            svc_age = cfg['age'].get('by_service', {}).get(service, {'mean': 32, 'std': 8})
            age = int(np.clip(self.rng.normal(svc_age['mean'], svc_age['std']), 18, 55))
            los = self._gen_los(age)
        
        age_entry = max(18, age - los)
        
        # Ethnicity
        ethnicity = self._gen_ethnicity(service)
        
        # Rank
        rank = self._gen_rank(los)
        
        # JMES
        jmes_dist = cfg['jmes']['new_joiner_distribution'] if is_new else cfg['jmes']['baseline_distribution']
        jmes = self._choice_dict(jmes_dist)
        
        # Health risk
        health = 'Low' if age < 35 else ('Medium' if age < 45 else 'High')
        
        # Priors
        prior_inj = 0 if is_new else int(self.rng.poisson(los * 0.25 * trade_risk.get('injury_multiplier', 1)))
        prior_dep = 0 if is_new else int(self.rng.poisson(los * 0.3))
        prior_preg = 0
        if gender == Gender.FEMALE and not is_new:
            prior_preg = int(self.rng.poisson(min(los, age - 18) * 0.05))
        
        return EnhancedPersonnelMaster(
            service_branch=ServiceBranch(service),
            age_at_entry=age_entry,
            age_start=age,
            gender=gender,
            ethnicity=Ethnicity(ethnicity),
            rank=RankBand(rank),
            trade=Trade(trade),
            length_of_service_start=los,
            engagement_type=EngagementType.OPEN,
            baseline_jmes=JMESStatus(jmes),
            baseline_health_risk=HealthRisk(health),
            unit_env_type=UnitEnvType.STANDARD,
            initial_deployability=jmes != 'MND',
            prior_injury_count=prior_inj,
            prior_deployment_count=prior_dep,
            prior_pregnancy_count=prior_preg,
            performance_band='average',
            cohort_month=cohort_month
        )
    
    def _gen_trade(self, gender: Gender) -> str:
        weights = self.trade_probs.copy()
        gender_rates = self.config['gender'].get('by_trade', {})
        for i, t in enumerate(self.trade_opts):
            gr = gender_rates.get(t, 0.15)
            weights[i] *= (gr if gender == Gender.FEMALE else (1 - gr))
        weights /= weights.sum()
        return self._choice(self.trade_opts, weights)
    
    def _gen_los(self, age: int) -> int:
        max_los = age - 18
        bands = self.config['length_of_service']['bands']
        band = self._choice_dict(bands)
        if '-' in band:
            lo, hi = map(int, band.split('-'))
        elif '+' in band:
            lo, hi = int(band.replace('+', '')), 37
        else:
            lo, hi = 0, 4
        lo, hi = min(lo, max_los), min(hi, max_los)
        return self.rng.integers(lo, max(lo + 1, hi + 1)) if hi > lo else lo
    
    def _gen_ethnicity(self, service: str) -> str:
        mods = self.config.get('ethnicity', {}).get('service_modifiers', {}).get(service, {})
        probs = self.eth_probs.copy()
        for i, e in enumerate(self.eth_opts):
            probs[i] *= mods.get(e, 1.0)
        probs /= probs.sum()
        return self._choice(self.eth_opts, probs)
    
    def _gen_rank(self, los: int) -> str:
        ranks = self.config.get('rank', {}).get('bands', {})
        min_los = self.config.get('rank', {}).get('min_los_for_rank', {})
        eligible = [(r, p) for r, p in ranks.items() if los >= min_los.get(r, 0)]
        if not eligible:
            return 'OR2'
        opts, probs = zip(*eligible)
        probs = np.array(probs)
        probs /= probs.sum()
        return self._choice(list(opts), probs)
    
    def generate_initial_cohort(self, n: int) -> pd.DataFrame:
        print(f"Generating {n:,} personnel...")
        for i in range(n):
            p = self.generate_person(is_new=False, cohort_month=0)
            self.master_records.append(p.to_dict())
            self.active_personnel[p.person_id] = {
                'master': p, 'jmes': p.baseline_jmes, 'rank': p.rank,
                'inj_hist': [], 'tot_inj': p.prior_injury_count, 'inj_12': [],
                'deploy': None, 'deploy_tot': p.prior_deployment_count * 4,
                'preg': None, 'preg_cnt': p.prior_pregnancy_count,
                'recovery': 0, 'surv': 0.0, 'jmes_event': False,
                'chronic': p.chronic_conditions.copy(), 'los': p.length_of_service_start,
                'age': p.age_start
            }
            if (i + 1) % 25000 == 0:
                print(f"  {i + 1:,} / {n:,}")
                gc.collect()
        
        self._verify_cohort()
        return pd.DataFrame(self.master_records)
    
    def _verify_cohort(self):
        df = pd.DataFrame(self.master_records)
        print("\n=== VERIFICATION ===")
        print(f"✓ Total: {len(df):,}")
        print(f"✓ Army: {(df['service_branch'] == 'Army').mean():.1%}")
        print(f"✓ Female: {(df['gender'] == 'Female').mean():.1%}")
        print(f"✓ MFD: {(df['baseline_jmes'] == 'MFD').mean():.1%}")
        print(f"✓ Mean age: {df['age_start'].mean():.1f}")
        print(f"✓ Mean LoS: {df['length_of_service_start'].mean():.1f}")
        print("Ethnicity breakdown:")
        print(df['ethnicity'].value_counts(normalize=True).head(5))
        print("=" * 40)
    
    def simulate_month(self, month: int) -> Tuple[List[Dict], List[Dict]]:
        cfg = self.config
        month_recs, inj_recs = [], []
        outflows = []
        
        for pid, s in self.active_personnel.items():
            m = s['master']
            s['age'] += 1/12
            s['los'] += 1/12
            s['inj_12'] = [x for x in s['inj_12'] if month - x < 12]
            
            # Deployment
            deploy = self._sim_deploy(s, m, month)
            
            # Training
            train = self._sim_train(s)
            
            # Injury
            inj = self._sim_injury(s, m, deploy, train, month)
            if inj:
                inj_recs.append(inj)
                s['tot_inj'] += 1
                s['inj_12'].append(month)
            
            # JMES
            jmes_evt, jmes_dir = self._sim_jmes(s, m, inj)
            
            # Pregnancy
            preg_stat, preg_mo = self._sim_preg(s, m, month)
            
            # Chronic
            chronic_new = self._sim_chronic(s, inj)
            
            # Sick days
            sick = self._calc_sick(s, inj, preg_stat)
            
            # Outflow
            out, out_reason = self._sim_outflow(s, m, month)
            
            s['surv'] += 1/12
            
            rec = EnhancedPersonMonth(
                person_id=pid, month=month, age=s['age'], current_los=s['los'],
                current_rank=s['rank'], jmes_current=s['jmes'],
                jmes_event_this_month=jmes_evt, jmes_direction=jmes_dir,
                injury_occurred=inj is not None,
                injury_type=InjuryType(inj['type']) if inj else InjuryType.NONE,
                injury_body_region=BodyRegion(inj['region']) if inj else BodyRegion.OTHER,
                injury_severity=inj['severity'] if inj else 0,
                injury_is_recurrence=inj['recur'] if inj else False,
                injury_context=inj['ctx'] if inj else 'routine',
                total_injury_count=s['tot_inj'],
                injuries_last_12_months=len(s['inj_12']),
                active_recovery=s['recovery'] > 0,
                recovery_days_remaining=s['recovery'],
                deployment_status=deploy,
                total_deployment_months=s['deploy_tot'],
                training_phase=train,
                pregnancy_status=preg_stat,
                pregnancy_month=preg_mo,
                total_pregnancies=s['preg_cnt'],
                sick_days=sick,
                outflow_flag=out,
                outflow_reason=out_reason,
                survival_time=s['surv'],
                event_flag=jmes_evt,
                censor_flag=out and not s['jmes_event'],
                new_chronic_condition=chronic_new is not None,
                chronic_condition_type=chronic_new or ChronicCondition.NONE
            )
            month_recs.append(rec.to_dict())
            
            if out:
                outflows.append(pid)
                self.stats['outflows'] += 1
        
        for pid in outflows:
            del self.active_personnel[pid]
        
        # Inflow
        n_in = len(outflows) + self.rng.integers(-20, 21)
        for _ in range(max(0, n_in)):
            p = self.generate_person(is_new=True, cohort_month=month)
            self.master_records.append(p.to_dict())
            self.active_personnel[p.person_id] = {
                'master': p, 'jmes': p.baseline_jmes, 'rank': p.rank,
                'inj_hist': [], 'tot_inj': 0, 'inj_12': [],
                'deploy': None, 'deploy_tot': 0,
                'preg': None, 'preg_cnt': 0,
                'recovery': 0, 'surv': 0.0, 'jmes_event': False,
                'chronic': [], 'los': 0, 'age': p.age_start
            }
            rec = EnhancedPersonMonth(
                person_id=p.person_id, month=month, age=p.age_start,
                current_los=0, current_rank=p.rank, jmes_current=p.baseline_jmes,
                inflow_flag=True, pregnancy_status=PregnancyStatus.NOT_APPLICABLE if p.gender == Gender.MALE else PregnancyStatus.NOT_PREGNANT
            )
            month_recs.append(rec.to_dict())
            self.stats['inflows'] += 1
        
        return month_recs, inj_recs
    
    def _sim_deploy(self, s, m, month) -> DeploymentType:
        if s.get('preg'):
            return DeploymentType.NOT_DEPLOYED
        if s['jmes'] == JMESStatus.MND:
            return DeploymentType.NOT_DEPLOYED
        if s['deploy']:
            s['deploy']['mo'] += 1
            if s['deploy']['mo'] >= s['deploy']['dur']:
                s['deploy_tot'] += s['deploy']['dur']
                s['deploy'] = None
                return DeploymentType.NOT_DEPLOYED
            return DeploymentType(s['deploy']['type'])
        
        rate = 0.065 * self.config.get('deployment', {}).get('service_rates', {}).get(m.service_branch.value, 1)
        if self.rng.random() < rate:
            dur = self.rng.integers(2, 7)
            s['deploy'] = {'type': 'Operational_low_intensity', 'dur': dur, 'mo': 1}
            return DeploymentType.OPERATIONAL_LOW
        return DeploymentType.NOT_DEPLOYED
    
    def _sim_train(self, s) -> TrainingPhase:
        if s.get('deploy') or s.get('preg'):
            return TrainingPhase.NONE
        if self.rng.random() < 0.12:
            return self._choice([TrainingPhase.SPECIALIST, TrainingPhase.EXERCISE_MAJOR, TrainingPhase.EXERCISE_MINOR],
                               np.array([0.4, 0.3, 0.3]))
        return TrainingPhase.NONE
    
    def _sim_injury(self, s, m, deploy, train, month) -> Optional[Dict]:
        base = 0.025
        trade_mult = self.config.get('trades', {}).get('risk_profiles', {}).get(m.trade.value, {}).get('injury_multiplier', 1)
        age_mult = 1.0 + max(0, (s['age'] - 35) * 0.02)
        ctx_mult = 1.6 if deploy != DeploymentType.NOT_DEPLOYED else (1.8 if train != TrainingPhase.NONE else 1.0)
        vuln_mult = 1.0 + 0.1 * min(5, s['tot_inj'])
        
        p = min(0.15, base * trade_mult * age_mult * ctx_mult * vuln_mult)
        if self.rng.random() >= p:
            s['recovery'] = max(0, s['recovery'] - 30)
            return None
        
        self.stats['injuries'] += 1
        itype = self._choice(self.inj_opts, self.inj_probs)
        region = self._choice(self.body_opts, self.body_probs)
        recur = region in [r for (mo, r) in s['inj_hist']]
        sev = self.rng.integers(1, 8) + (1 if recur else 0)
        s['inj_hist'].append((month, region))
        s['recovery'] = self.rng.integers(7, 60)
        
        ctx = 'deployment' if deploy != DeploymentType.NOT_DEPLOYED else ('training' if train != TrainingPhase.NONE else 'routine')
        
        return {'type': itype, 'region': region, 'severity': min(10, sev), 'recur': recur, 'ctx': ctx}
    
    def _sim_jmes(self, s, m, inj) -> Tuple[bool, str]:
        trans = self.config['jmes']['base_transitions']
        cur = s['jmes']
        
        age_mod = 1.0 + max(0, (s['age'] - 40) * 0.02)
        inj_mod = 1.0 + 0.15 * len(s['inj_12'])
        
        if cur == JMESStatus.MFD:
            p_mld = trans['MFD_to_MLD'] * age_mod * inj_mod
            p_mnd = trans['MFD_to_MND'] * age_mod * inj_mod
            if inj and inj['severity'] >= 7:
                p_mld *= 3
            r = self.rng.random()
            if r < p_mnd:
                s['jmes'] = JMESStatus.MND
                s['jmes_event'] = True
                self.stats['jmes_det'] += 1
                return True, 'deterioration'
            if r < p_mnd + p_mld:
                s['jmes'] = JMESStatus.MLD
                s['jmes_event'] = True
                self.stats['jmes_det'] += 1
                return True, 'deterioration'
        elif cur == JMESStatus.MLD:
            p_mnd = trans['MLD_to_MND'] * age_mod
            p_mfd = trans['MLD_to_MFD'] / age_mod
            r = self.rng.random()
            if r < p_mnd:
                s['jmes'] = JMESStatus.MND
                s['jmes_event'] = True
                return True, 'deterioration'
            if r < p_mnd + p_mfd:
                s['jmes'] = JMESStatus.MFD
                return False, 'improvement'
        else:
            p_mld = trans['MND_to_MLD']
            if self.rng.random() < p_mld:
                s['jmes'] = JMESStatus.MLD
                return False, 'improvement'
        return False, 'stable'
    
    def _sim_preg(self, s, m, month) -> Tuple[PregnancyStatus, int]:
        if m.gender == Gender.MALE:
            return PregnancyStatus.NOT_APPLICABLE, 0
        if s.get('preg'):
            s['preg']['mo'] += 1
            mo = s['preg']['mo']
            if mo > 15:  # 9 preg + 6 leave
                s['preg'] = None
                return PregnancyStatus.NOT_PREGNANT, 0
            if mo <= 3:
                return PregnancyStatus.T1, mo
            if mo <= 6:
                return PregnancyStatus.T2, mo
            if mo <= 9:
                return PregnancyStatus.T3, mo
            return PregnancyStatus.MATERNITY_LEAVE, mo
        
        age = s['age']
        if age > 45 or s['jmes'] == JMESStatus.MND:
            return PregnancyStatus.NOT_PREGNANT, 0
        
        rates = self.config.get('pregnancy', {}).get('conception_rates_by_age', {})
        annual_rate = 0.05
        for band, rate in rates.items():
            if '+' in band:
                lo = int(band.replace('+', ''))
                hi = 99
            else:
                parts = band.split('-')
                lo, hi = int(parts[0]), int(parts[1])
            if lo <= age <= hi:
                annual_rate = rate
                break
        
        if self.rng.random() < annual_rate / 12:
            s['preg'] = {'mo': 1, 'start': month}
            s['preg_cnt'] += 1
            self.stats['pregnancies'] += 1
            return PregnancyStatus.T1, 1
        
        return PregnancyStatus.NOT_PREGNANT, 0
    
    def _sim_chronic(self, s, inj) -> Optional[ChronicCondition]:
        if s['tot_inj'] >= 5 and self.rng.random() < 0.03:
            cond = self._choice([ChronicCondition.CHRONIC_PAIN, ChronicCondition.MUSCULOSKELETAL],
                               np.array([0.6, 0.4]))
            s['chronic'].append(cond)
            return cond
        return None
    
    def _calc_sick(self, s, inj, preg) -> int:
        base = 0
        if inj:
            base = min(20, inj['severity'] * 2)
        if preg == PregnancyStatus.T3:
            base += self.rng.integers(0, 5)
        if s['recovery'] > 0:
            base += 2
        return min(30, base)
    
    def _sim_outflow(self, s, m, month) -> Tuple[bool, Optional[OutflowReason]]:
        cfg = self.config['turnover']
        p = cfg['monthly_rate']
        
        # LoS weight
        los = s['los']
        if los < 3:
            p *= 1.4
        elif los < 5:
            p *= 1.2
        elif los > 20:
            p *= 1.1
        
        # JMES weight
        if s['jmes'] == JMESStatus.MND:
            p *= 2.5
        elif s['jmes'] == JMESStatus.MLD:
            p *= 1.3
        
        # Age
        if s['age'] > 50:
            p *= 1.5
        
        if self.rng.random() < p:
            if s['jmes'] == JMESStatus.MND:
                reason = OutflowReason.MEDICAL_DISCHARGE
            elif s['age'] > 50:
                reason = OutflowReason.RETIREMENT
            else:
                reason = self._choice(
                    [OutflowReason.VOLUNTARY_EARLY, OutflowReason.END_OF_ENGAGEMENT, OutflowReason.CAREER_CHANGE],
                    np.array([0.4, 0.35, 0.25])
                )
            return True, reason
        return False, None
    
    def run_simulation(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        cfg = self.config['population']
        n = cfg['initial_size']
        months = cfg['simulation_months']
        
        master_df = self.generate_initial_cohort(n)
        
        print(f"\nSimulating {months} months...")
        for mo in range(1, months + 1):
            mrecs, irecs = self.simulate_month(mo)
            self.month_records.extend(mrecs)
            self.injury_records.extend(irecs)
            
            if mo % 12 == 0:
                yr = mo // 12
                pop = len(self.active_personnel)
                mfd = sum(1 for s in self.active_personnel.values() if s['jmes'] == JMESStatus.MFD) / max(1, pop)
                print(f"  Year {yr}: Pop={pop:,}, MFD={mfd:.1%}, Injuries={self.stats['injuries']:,}")
        
        master_df = pd.DataFrame(self.master_records)
        month_df = pd.DataFrame(self.month_records)
        injury_df = pd.DataFrame(self.injury_records) if self.injury_records else pd.DataFrame()
        
        self._final_report()
        return master_df, month_df, injury_df
    
    def _final_report(self):
        print("\n" + "=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)
        print(f"Total personnel (master): {len(self.master_records):,}")
        print(f"Monthly records: {len(self.month_records):,}")
        print(f"Injury events: {len(self.injury_records):,}")
        print(f"Total outflows: {self.stats['outflows']:,}")
        print(f"Total inflows: {self.stats['inflows']:,}")
        print(f"JMES deteriorations: {self.stats['jmes_det']:,}")
        print(f"Pregnancies: {self.stats['pregnancies']:,}")
        print("=" * 60)


if __name__ == "__main__":
    gen = EnhancedSyntheticGenerator('configs/enhanced_config.yaml', seed=42)
    # Small test
    gen.config['population']['initial_size'] = 5000
    gen.config['population']['simulation_months'] = 24
    master, monthly, injuries = gen.run_simulation()
    print(f"\nMaster: {master.shape}, Monthly: {monthly.shape}, Injuries: {injuries.shape}")

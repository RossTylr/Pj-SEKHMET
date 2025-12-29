"""
JMES Synthetic Workforce - Data Generation Engine
==================================================
Implements probabilistic simulation with Chain-of-Verification.
Follows FAANG-level patterns for health data engineering.
"""

import numpy as np
import pandas as pd
import yaml
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import uuid
from dataclasses import asdict

from models import (
    PersonnelMaster, PersonMonth,
    ServiceBranch, ServiceType, Gender, RankBand, Trade,
    JMESStatus, InjuryType, DeploymentStatus, TrainingPhase,
    PregnancyStatus, EngagementType, UnitEnvType, HealthRisk,
    OutflowReason, PERSONNEL_MASTER_SCHEMA, PERSON_MONTH_SCHEMA,
    validate_dataframe, apply_schema
)


class SyntheticDataGenerator:
    """
    Main engine for generating synthetic Defence workforce data.
    
    Design Principles:
    1. Reproducibility via seed control
    2. Chain-of-Verification at each step
    3. Configurable via YAML
    4. Memory-efficient chunked processing
    """
    
    def __init__(self, config_path: str, seed: int = 42):
        self.config = self._load_config(config_path)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # State tracking
        self.master_records: List[Dict] = []
        self.month_records: List[Dict] = []
        self.active_personnel: Dict[str, Dict] = {}  # person_id -> current state
        
        # Verification counters
        self.verification_log = {
            'total_inflows': 0,
            'total_outflows': 0,
            'jmes_deteriorations': 0,
            'injuries': 0,
            'pregnancies': 0
        }
    
    def _load_config(self, config_path: str) -> dict:
        """Load and validate configuration"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Verification: Check required keys
        required = ['population', 'turnover', 'service_mix', 'jmes']
        missing = [k for k in required if k not in config]
        if missing:
            raise ValueError(f"Missing config keys: {missing}")
        
        return config
    
    # ============================================================
    # PERSONNEL GENERATION
    # ============================================================
    
    def generate_person(self, is_new_joiner: bool = False) -> PersonnelMaster:
        """Generate a single synthetic person with realistic attributes"""
        cfg = self.config
        
        # Service branch
        service = self.rng.choice(
            list(cfg['service_mix'].keys()),
            p=list(cfg['service_mix'].values())
        )
        
        # Gender (adjusted by trade)
        trade = self.rng.choice(
            list(cfg['trades'].keys()),
            p=list(cfg['trades'].values())
        )
        female_rate = cfg['gender']['trade_adjustments'].get(
            trade, cfg['gender']['overall_female_rate']
        )
        gender = Gender.FEMALE if self.rng.random() < female_rate else Gender.MALE
        
        # Age distribution
        if is_new_joiner:
            age_cfg = cfg['age']['inflow']
            age = int(np.clip(
                self.rng.normal(age_cfg['mean'], age_cfg['std']),
                age_cfg['min'], age_cfg['max']
            ))
            los = self.rng.integers(0, 3)  # 0-2 years for new joiners
        else:
            age = int(np.clip(
                self.rng.normal(cfg['age']['mean'], cfg['age']['std']),
                cfg['age']['min'], cfg['age']['max']
            ))
            # Length of service (correlated with age)
            los_band = self.rng.choice(
                ['0-4', '5-12', '13-25'],
                p=[cfg['length_of_service']['bands'][b] for b in ['0-4', '5-12', '13-25']]
            )
            if los_band == '0-4':
                los = self.rng.integers(0, 5)
            elif los_band == '5-12':
                los = self.rng.integers(5, 13)
            else:
                # 13-25 band - ensure valid range
                max_los = min(26, age - 17)
                if max_los > 13:
                    los = self.rng.integers(13, max_los)
                else:
                    los = self.rng.integers(5, 13)  # Fall back to mid-career
        
        # Ensure LoS is valid given age
        los = min(los, age - 18)
        
        # Rank band (correlated with LoS)
        if los < 5:
            rank_weights = [0.7, 0.2, 0.05, 0.04, 0.01]
        elif los < 13:
            rank_weights = [0.3, 0.4, 0.15, 0.10, 0.05]
        else:
            rank_weights = [0.1, 0.3, 0.25, 0.20, 0.15]
        
        rank_options = [RankBand.OR2_OR4, RankBand.OR5_OR7, RankBand.OR8_OR9, 
                        RankBand.OF1_OF3, RankBand.OF4_OF5]
        rank_idx = self.rng.choice(len(rank_options), p=rank_weights)
        rank = rank_options[rank_idx]
        
        # Baseline JMES (new joiners healthier)
        if is_new_joiner:
            jmes_probs = [0.95, 0.04, 0.01]
        else:
            jmes_probs = [
                cfg['jmes']['baseline_distribution']['MFD'],
                cfg['jmes']['baseline_distribution']['MLD'],
                cfg['jmes']['baseline_distribution']['MND']
            ]
        
        jmes_options = [JMESStatus.MFD, JMESStatus.MLD, JMESStatus.MND]
        jmes_idx = self.rng.choice(len(jmes_options), p=jmes_probs)
        baseline_jmes = jmes_options[jmes_idx]
        
        # Other attributes
        engagement = self.rng.choice(
            list(cfg['engagement_types'].keys()),
            p=list(cfg['engagement_types'].values())
        )
        
        unit_env = self.rng.choice(
            list(cfg['unit_environment'].keys()),
            p=list(cfg['unit_environment'].values())
        )
        
        # Health risk (correlated with age and trade)
        risk_modifier = (age - 30) * 0.02 + (0.1 if trade == 'CMT' else 0)
        risk_probs = np.softmax([
            0.6 - risk_modifier,
            0.3,
            0.1 + risk_modifier
        ])
        risk_options = [HealthRisk.LOW, HealthRisk.MEDIUM, HealthRisk.HIGH]
        risk_probs_norm = np.clip(risk_probs, 0.01, 0.98)
        risk_probs_norm = risk_probs_norm / risk_probs_norm.sum()
        risk_idx = self.rng.choice(len(risk_options), p=risk_probs_norm)
        health_risk = risk_options[risk_idx]
        
        return PersonnelMaster(
            person_id=str(uuid.uuid4()),
            service_branch=ServiceBranch(service),
            regular_reserve=ServiceType.REGULAR,
            age_start=age,
            gender=gender,
            rank_band=rank,
            trade=Trade(trade),
            length_of_service_start=los,
            engagement_type=EngagementType(engagement),
            baseline_jmes=baseline_jmes,
            baseline_health_risk=health_risk,
            unit_env_type=UnitEnvType(unit_env),
            initial_deployability=baseline_jmes != JMESStatus.MND
        )
    
    def generate_initial_cohort(self, n: int) -> pd.DataFrame:
        """Generate the initial population at t=0"""
        print(f"Generating initial cohort of {n:,} personnel...")
        
        records = []
        for i in range(n):
            person = self.generate_person(is_new_joiner=False)
            records.append(person.to_dict())
            
            # Track active personnel
            self.active_personnel[person.person_id] = {
                'master': person,
                'current_jmes': person.baseline_jmes,
                'pregnancy_month': 0,  # 0 = not pregnant
                'survival_clock': 0.0,
                'had_jmes_event': False
            }
            
            if (i + 1) % 10000 == 0:
                print(f"  Generated {i+1:,} / {n:,}")
        
        self.master_records = records
        df = pd.DataFrame(records)
        
        # Chain-of-Verification
        self._verify_initial_cohort(df)
        
        return apply_schema(df, PERSONNEL_MASTER_SCHEMA)
    
    def _verify_initial_cohort(self, df: pd.DataFrame):
        """Verify initial cohort meets specifications"""
        cfg = self.config
        
        print("\n=== Chain-of-Verification: Initial Cohort ===")
        
        # Service mix
        service_dist = df['service_branch'].value_counts(normalize=True)
        for service, expected in cfg['service_mix'].items():
            actual = service_dist.get(service, 0)
            status = "✓" if abs(actual - expected) < 0.03 else "⚠"
            print(f"  {status} {service}: {actual:.1%} (expected {expected:.1%})")
        
        # Gender
        female_rate = (df['gender'] == 'Female').mean()
        expected_female = cfg['gender']['overall_female_rate']
        status = "✓" if abs(female_rate - expected_female) < 0.03 else "⚠"
        print(f"  {status} Female rate: {female_rate:.1%} (expected ~{expected_female:.1%})")
        
        # Age
        age_mean = df['age_start'].mean()
        status = "✓" if 30 <= age_mean <= 35 else "⚠"
        print(f"  {status} Mean age: {age_mean:.1f} (expected 31-34)")
        
        # JMES baseline
        mfd_rate = (df['baseline_jmes'] == 'MFD').mean()
        status = "✓" if mfd_rate >= 0.80 else "⚠"
        print(f"  {status} MFD rate: {mfd_rate:.1%} (expected ≥80%)")
        
        print("=" * 50)
    
    # ============================================================
    # MONTHLY SIMULATION ENGINE
    # ============================================================
    
    def simulate_month(self, month: int) -> List[Dict]:
        """Simulate one month for all active personnel"""
        cfg = self.config
        month_records = []
        outflow_ids = []
        
        for person_id, state in self.active_personnel.items():
            master = state['master']
            
            # Calculate current age
            current_age = master.age_start + (month / 12)
            
            # ---- JMES Transition ----
            current_jmes, jmes_event = self._simulate_jmes_transition(
                state['current_jmes'], master, state
            )
            if jmes_event:
                self.verification_log['jmes_deteriorations'] += 1
            
            # ---- Injury Event ----
            injury_type, severity = self._simulate_injury(master, state)
            if injury_type != InjuryType.NONE:
                self.verification_log['injuries'] += 1
            
            # ---- Deployment ----
            deployment = self._simulate_deployment(master)
            
            # ---- Training ----
            training = self._simulate_training(master)
            
            # ---- Pregnancy (female only) ----
            pregnancy, new_preg_month = self._simulate_pregnancy(
                master, state, current_age
            )
            if pregnancy != PregnancyStatus.NOT_PREGNANT and state['pregnancy_month'] == 0:
                self.verification_log['pregnancies'] += 1
            
            # ---- Sick Days ----
            sick_days = self._calculate_sick_days(injury_type, pregnancy)
            
            # ---- Outflow Check ----
            outflow, outflow_reason = self._check_outflow(master, current_jmes)
            
            # ---- Update Survival Clock ----
            survival_time = state['survival_clock'] + (1/12)
            censor_flag = not state['had_jmes_event'] and (outflow or month == cfg['population']['simulation_months'])
            
            # Create month record
            pm = PersonMonth(
                person_id=person_id,
                month=month,
                age=current_age,
                jmes_current=current_jmes,
                jmes_event_this_month=jmes_event,
                injury_type=injury_type,
                injury_severity_score=severity,
                deployment_status=deployment,
                training_phase=training,
                pregnancy_status=pregnancy,
                sick_days=sick_days,
                outflow_flag=outflow,
                outflow_reason=outflow_reason if outflow else None,
                inflow_flag=False,
                survival_time=survival_time,
                censor_flag=censor_flag
            )
            month_records.append(pm.to_dict())
            
            # Update state
            state['current_jmes'] = current_jmes
            state['pregnancy_month'] = new_preg_month
            state['survival_clock'] = survival_time
            if jmes_event:
                state['had_jmes_event'] = True
            
            if outflow:
                outflow_ids.append(person_id)
                self.verification_log['total_outflows'] += 1
        
        # Remove outflow personnel
        for pid in outflow_ids:
            del self.active_personnel[pid]
        
        # ---- Inflow (new joiners) ----
        inflow_records = self._generate_inflow(month, len(outflow_ids))
        month_records.extend(inflow_records)
        
        return month_records
    
    def _simulate_jmes_transition(
        self, current: JMESStatus, master: PersonnelMaster, state: Dict
    ) -> Tuple[JMESStatus, bool]:
        """Simulate JMES state transition"""
        trans = self.config['jmes']['transitions']
        
        # Risk modifiers
        age_modifier = max(0, (master.age_start + state['survival_clock'] - 35) * 0.001)
        trade_modifier = 0.002 if master.trade in [Trade.CMT, Trade.PARAMEDIC] else 0
        
        if current == JMESStatus.MFD:
            p_mld = trans['MFD_to_MLD'] + age_modifier + trade_modifier
            p_mnd = trans['MFD_to_MND']
            
            roll = self.rng.random()
            if roll < p_mnd:
                return JMESStatus.MND, True
            elif roll < p_mnd + p_mld:
                return JMESStatus.MLD, True
            return JMESStatus.MFD, False
            
        elif current == JMESStatus.MLD:
            p_mnd = trans['MLD_to_MND']
            p_mfd = trans['MLD_to_MFD']
            
            roll = self.rng.random()
            if roll < p_mnd:
                return JMESStatus.MND, True
            elif roll < p_mnd + p_mfd:
                return JMESStatus.MFD, False  # Recovery, not deterioration
            return JMESStatus.MLD, False
            
        else:  # MND
            p_mld = trans['MND_to_MLD']
            p_mfd = trans['MND_to_MFD']
            
            roll = self.rng.random()
            if roll < p_mfd:
                return JMESStatus.MFD, False
            elif roll < p_mfd + p_mld:
                return JMESStatus.MLD, False
            return JMESStatus.MND, False
    
    def _simulate_injury(
        self, master: PersonnelMaster, state: Dict
    ) -> Tuple[InjuryType, int]:
        """Simulate monthly injury event"""
        cfg = self.config['injuries']
        
        # Base probability
        p_injury = cfg['baseline_monthly_mski']
        
        # Trade multiplier
        trade_mult = cfg['trade_multipliers'].get(master.trade.value, 1.0)
        p_injury *= trade_mult
        
        # Environment multiplier
        if master.unit_env_type == UnitEnvType.HIGH_RISK:
            p_injury *= 1.3
        
        if self.rng.random() < p_injury:
            # Determine injury type
            injury_options = [InjuryType.MSKI_MINOR, InjuryType.MSKI_MAJOR, 
                             InjuryType.MH_EPISODE, InjuryType.OTHER]
            injury_idx = self.rng.choice(len(injury_options), p=[0.50, 0.25, 0.15, 0.10])
            injury_type = injury_options[injury_idx]
            
            # Severity score (1-10)
            if injury_type == InjuryType.MSKI_MINOR:
                severity = self.rng.integers(1, 4)
            elif injury_type == InjuryType.MSKI_MAJOR:
                severity = self.rng.integers(5, 9)
            else:
                severity = self.rng.integers(3, 7)
            
            return injury_type, severity
        
        return InjuryType.NONE, 0
    
    def _simulate_deployment(self, master: PersonnelMaster) -> DeploymentStatus:
        """Simulate deployment status"""
        cfg = self.config['deployment']
        
        base_rate = cfg['baseline_monthly_rate']
        service_mult = cfg['service_rates'].get(master.service_branch.value, 1.0)
        
        if self.rng.random() < base_rate * service_mult:
            deploy_options = [DeploymentStatus.LOW_TEMPO, DeploymentStatus.HIGH_TEMPO]
            deploy_idx = self.rng.choice(len(deploy_options), 
                                         p=[cfg['intensity']['Low_tempo'], cfg['intensity']['High_tempo']])
            return deploy_options[deploy_idx]
        
        return DeploymentStatus.NOT_DEPLOYED
    
    def _simulate_training(self, master: PersonnelMaster) -> TrainingPhase:
        """Simulate training phase"""
        cfg = self.config['training']
        
        if self.rng.random() < cfg['monthly_rate']:
            train_options = [TrainingPhase.LOW_RISK, TrainingPhase.HIGH_RISK]
            train_idx = self.rng.choice(len(train_options),
                                        p=[cfg['intensity']['Low_risk'], cfg['intensity']['High_risk']])
            return train_options[train_idx]
        
        return TrainingPhase.NONE
    
    def _simulate_pregnancy(
        self, master: PersonnelMaster, state: Dict, current_age: float
    ) -> Tuple[PregnancyStatus, int]:
        """Simulate pregnancy status for females 18-45"""
        if master.gender != Gender.FEMALE or current_age > 45:
            return PregnancyStatus.NOT_PREGNANT, 0
        
        cfg = self.config['pregnancy']
        current_preg_month = state['pregnancy_month']
        
        if current_preg_month == 0:
            # Not pregnant - check for conception
            if self.rng.random() < cfg['monthly_conception_rate']:
                return PregnancyStatus.T1, 1
            return PregnancyStatus.NOT_PREGNANT, 0
        
        # Already pregnant/postpartum - advance state
        new_month = current_preg_month + 1
        total_duration = cfg['duration']['pregnancy_months'] + cfg['duration']['postpartum_months']
        
        if new_month > total_duration:
            return PregnancyStatus.NOT_PREGNANT, 0
        elif new_month <= 3:
            return PregnancyStatus.T1, new_month
        elif new_month <= 6:
            return PregnancyStatus.T2, new_month
        elif new_month <= 9:
            return PregnancyStatus.T3, new_month
        else:
            return PregnancyStatus.POSTPARTUM, new_month
    
    def _calculate_sick_days(
        self, injury: InjuryType, pregnancy: PregnancyStatus
    ) -> int:
        """Calculate sick days based on injury and pregnancy"""
        base = 0
        
        if injury == InjuryType.MSKI_MINOR:
            base = self.rng.integers(1, 5)
        elif injury == InjuryType.MSKI_MAJOR:
            base = self.rng.integers(5, 20)
        elif injury == InjuryType.MH_EPISODE:
            base = self.rng.integers(3, 15)
        elif injury == InjuryType.OTHER:
            base = self.rng.integers(1, 10)
        
        # Pregnancy adjustments
        if pregnancy == PregnancyStatus.T3:
            base += self.rng.integers(0, 5)
        elif pregnancy == PregnancyStatus.POSTPARTUM:
            base += self.rng.integers(2, 10)
        
        return min(base, 30)
    
    def _check_outflow(
        self, master: PersonnelMaster, jmes: JMESStatus
    ) -> Tuple[bool, Optional[OutflowReason]]:
        """Check if personnel exits service this month"""
        cfg = self.config['turnover']
        
        # Base monthly outflow probability
        p_outflow = cfg['monthly_rate']
        
        # LoS weighting (optional)
        los = master.length_of_service_start
        if los < 5:
            p_outflow *= cfg['los_weights'].get('0-4', 1.0)
        elif los < 13:
            p_outflow *= cfg['los_weights'].get('5-12', 1.0)
        else:
            p_outflow *= cfg['los_weights'].get('13+', 1.0)
        
        # MND increases medical discharge probability
        if jmes == JMESStatus.MND:
            p_outflow *= 2.0
        
        if self.rng.random() < p_outflow:
            # Determine reason
            if jmes == JMESStatus.MND and self.rng.random() < 0.5:
                reason = OutflowReason.MEDICAL_DISCHARGE
            else:
                reason_options = [
                    OutflowReason.PREMATURE_VOL,
                    OutflowReason.END_ENGAGEMENT,
                    OutflowReason.NORMAL_TERM,
                    OutflowReason.ADMINISTRATIVE
                ]
                reason_idx = self.rng.choice(len(reason_options), p=[0.35, 0.30, 0.25, 0.10])
                reason = reason_options[reason_idx]
            return True, reason
        
        return False, None
    
    def _generate_inflow(self, month: int, n_outflow: int) -> List[Dict]:
        """Generate new joiners to replace outflow"""
        # Target: maintain steady-state with noise
        n_inflow = n_outflow + self.rng.integers(-50, 51)
        n_inflow = max(0, n_inflow)
        
        records = []
        for _ in range(n_inflow):
            person = self.generate_person(is_new_joiner=True)
            
            # Add to master records
            self.master_records.append(person.to_dict())
            
            # Track active personnel
            self.active_personnel[person.person_id] = {
                'master': person,
                'current_jmes': person.baseline_jmes,
                'pregnancy_month': 0,
                'survival_clock': 0.0,
                'had_jmes_event': False
            }
            
            # Create first month record (with inflow flag)
            pm = PersonMonth(
                person_id=person.person_id,
                month=month,
                age=person.age_start,
                jmes_current=person.baseline_jmes,
                jmes_event_this_month=False,
                injury_type=InjuryType.NONE,
                injury_severity_score=0,
                deployment_status=DeploymentStatus.NOT_DEPLOYED,
                training_phase=TrainingPhase.HIGH_RISK,  # Initial training
                pregnancy_status=PregnancyStatus.NOT_PREGNANT,
                sick_days=0,
                outflow_flag=False,
                inflow_flag=True,
                survival_time=0.0,
                censor_flag=False
            )
            records.append(pm.to_dict())
            self.verification_log['total_inflows'] += 1
        
        return records
    
    # ============================================================
    # MAIN SIMULATION RUNNER
    # ============================================================
    
    def run_simulation(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run full simulation and return master + longitudinal tables"""
        cfg = self.config
        n_months = cfg['population']['simulation_months']
        n_personnel = cfg['population']['initial_size']
        
        # Step 1: Generate initial cohort
        master_df = self.generate_initial_cohort(n_personnel)
        
        # Step 2: Generate month 1 records for initial cohort
        print(f"\nRunning {n_months}-month simulation...")
        all_month_records = []
        
        for month in range(1, n_months + 1):
            month_records = self.simulate_month(month)
            all_month_records.extend(month_records)
            
            if month % 12 == 0:
                year = month // 12
                pop_size = len(self.active_personnel)
                mfd_rate = sum(
                    1 for s in self.active_personnel.values() 
                    if s['current_jmes'] == JMESStatus.MFD
                ) / max(pop_size, 1)
                print(f"  Year {year}: Pop={pop_size:,}, MFD={mfd_rate:.1%}")
        
        # Step 3: Create final DataFrames
        master_df = pd.DataFrame(self.master_records)
        month_df = pd.DataFrame(all_month_records)
        
        # Apply schemas
        master_df = apply_schema(master_df, PERSONNEL_MASTER_SCHEMA)
        month_df = apply_schema(month_df, PERSON_MONTH_SCHEMA)
        
        # Final verification
        self._final_verification(master_df, month_df)
        
        return master_df, month_df
    
    def _final_verification(self, master_df: pd.DataFrame, month_df: pd.DataFrame):
        """Final Chain-of-Verification"""
        cfg = self.config
        
        print("\n" + "=" * 60)
        print("FINAL CHAIN-OF-VERIFICATION")
        print("=" * 60)
        
        # Population stability
        initial_pop = cfg['population']['initial_size']
        final_pop = len(self.active_personnel)
        pop_change = abs(final_pop - initial_pop) / initial_pop
        status = "✓" if pop_change < cfg['validation']['population_stability'] else "⚠"
        print(f"{status} Population stability: {initial_pop:,} → {final_pop:,} ({pop_change:.1%} change)")
        
        # Turnover rate
        n_months = cfg['population']['simulation_months']
        avg_monthly_outflow = self.verification_log['total_outflows'] / n_months
        implied_annual = 1 - (1 - avg_monthly_outflow / initial_pop) ** 12
        status = "✓" if abs(implied_annual - 0.10) < cfg['validation']['turnover_tolerance'] else "⚠"
        print(f"{status} Implied annual turnover: {implied_annual:.1%} (target: 10%)")
        
        # JMES deteriorations
        print(f"ℹ JMES deterioration events: {self.verification_log['jmes_deteriorations']:,}")
        print(f"ℹ Total injuries: {self.verification_log['injuries']:,}")
        print(f"ℹ Total pregnancies: {self.verification_log['pregnancies']:,}")
        
        # Table sizes
        print(f"\n📊 Master table: {len(master_df):,} rows")
        print(f"📊 Longitudinal table: {len(month_df):,} rows")
        print(f"📊 Memory: Master={master_df.memory_usage(deep=True).sum()/1e6:.1f}MB, "
              f"Monthly={month_df.memory_usage(deep=True).sum()/1e6:.1f}MB")
        
        print("=" * 60)


# Utility for softmax
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

np.softmax = softmax  # Patch for numpy


if __name__ == "__main__":
    # Quick test
    generator = SyntheticDataGenerator(
        config_path="configs/simulation_config.yaml",
        seed=42
    )
    master_df, month_df = generator.run_simulation()
    
    print("\n✅ Simulation complete!")
    print(f"Master: {master_df.shape}")
    print(f"Monthly: {month_df.shape}")

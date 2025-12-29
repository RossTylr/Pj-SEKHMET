"""
JMES Synthetic Data Generator - Core simulation engine
"""

import yaml
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from models import (
    PersonnelMaster, PersonMonth, ServiceBranch, Gender, RankBand,
    Trade, JMESStatus, InjuryType, DeploymentStatus, TrainingPhase,
    PregnancyStatus, EngagementType, UnitEnvType, HealthRisk,
    OutflowReason, ServiceType
)


class SyntheticDataGenerator:
    """Generates synthetic personnel and monthly tracking data"""

    def __init__(self, config_path: str, seed: int = 42):
        """Initialize generator with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.rng = np.random.default_rng(seed)
        self.active_personnel = {}  # person_id -> state dict
        self.all_personnel = []  # All created personnel
        self.monthly_records = []

    def run_simulation(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run full simulation and return master and monthly dataframes"""
        print("=" * 60)
        print("JMES SIMULATION STARTING")
        print("=" * 60)

        # Generate initial cohort
        print("\n[1/3] Generating initial cohort...")
        self._generate_initial_cohort()
        print(f"  Created {len(self.active_personnel)} personnel")

        # Verify initial distributions
        self._verify_initial_cohort()

        # Run monthly simulation
        n_months = self.config['population']['simulation_months']
        print(f"\n[2/3] Running {n_months}-month simulation...")

        for month in range(1, n_months + 1):
            month_data = self._simulate_month(month)
            self.monthly_records.extend(month_data)

            if month % 12 == 0:
                year = month // 12
                pop = len(self.active_personnel)
                mfd_rate = sum(1 for s in self.active_personnel.values()
                              if s['jmes_current'] == JMESStatus.MFD) / pop
                print(f"  Year {year}: Pop={pop}, MFD={mfd_rate:.1%}")

        # Convert to DataFrames
        print("\n[3/3] Creating DataFrames...")
        master_df = pd.DataFrame([p.to_dict() for p in self.all_personnel])
        monthly_df = pd.DataFrame(self.monthly_records)

        print(f"  Master: {len(master_df):,} rows")
        print(f"  Monthly: {len(monthly_df):,} rows")

        # Final verification
        self._final_verification(master_df, monthly_df)

        print("\n" + "=" * 60)
        print("SIMULATION COMPLETE")
        print("=" * 60)

        return master_df, monthly_df

    def _generate_initial_cohort(self):
        """Generate initial population"""
        n = self.config['population']['initial_size']

        for _ in range(n):
            person = self._create_person()
            self.all_personnel.append(person)

            # Store active state
            self.active_personnel[person.person_id] = {
                'master': person,
                'jmes_current': person.baseline_jmes,
                'age': float(person.age_start),
                'los': person.length_of_service_start,
                'pregnancy_counter': 0,
                'survival_time': 0.0
            }

    def _create_person(self, is_inflow: bool = False) -> PersonnelMaster:
        """Create a single synthetic person"""
        # Service branch
        service_options = list(ServiceBranch)
        service_probs = [self.config['service_mix'][s.value] for s in service_options]
        service_idx = self.rng.choice(len(service_options), p=service_probs)
        service = service_options[service_idx]

        # Trade
        trade_options = list(Trade)
        trade_probs = [self.config['trades'][t.value] for t in trade_options]
        trade_idx = self.rng.choice(len(trade_options), p=trade_probs)
        trade = trade_options[trade_idx]

        # Gender (trade-adjusted)
        base_female_rate = self.config['gender']['overall_female_rate']
        trade_adjustment = self.config['gender']['trade_adjustments'].get(trade.value, base_female_rate)
        gender = Gender.FEMALE if self.rng.random() < trade_adjustment else Gender.MALE

        # Age
        if is_inflow:
            age_cfg = self.config['age']['inflow']
        else:
            age_cfg = self.config['age']

        age = int(np.clip(
            self.rng.normal(age_cfg['mean'], age_cfg['std']),
            age_cfg['min'], age_cfg['max']
        ))

        # Length of service
        los = self._generate_los(age, is_inflow)

        # Rank band
        rank_options = list(RankBand)
        rank_probs = [self.config['rank_bands'][r.value] for r in rank_options]
        rank_idx = self.rng.choice(len(rank_options), p=rank_probs)
        rank = rank_options[rank_idx]

        # Engagement type
        eng_options = list(EngagementType)
        eng_probs = [self.config['engagement_types'][e.value] for e in eng_options]
        eng_idx = self.rng.choice(len(eng_options), p=eng_probs)
        engagement = eng_options[eng_idx]

        # Unit environment
        unit_options = list(UnitEnvType)
        unit_probs = [self.config['unit_environment'][u.value] for u in unit_options]
        unit_idx = self.rng.choice(len(unit_options), p=unit_probs)
        unit_env = unit_options[unit_idx]

        # JMES baseline
        jmes_options = list(JMESStatus)
        jmes_probs = [self.config['jmes']['baseline_distribution'][j.value] for j in jmes_options]
        jmes_idx = self.rng.choice(len(jmes_options), p=jmes_probs)
        jmes = jmes_options[jmes_idx]

        # Health risk
        health_risk = HealthRisk.LOW

        return PersonnelMaster(
            service_branch=service,
            regular_reserve=ServiceType.REGULAR,
            age_start=age,
            gender=gender,
            rank_band=rank,
            trade=trade,
            length_of_service_start=los,
            engagement_type=engagement,
            baseline_jmes=jmes,
            baseline_health_risk=health_risk,
            unit_env_type=unit_env,
            initial_deployability=(jmes == JMESStatus.MFD)
        )

    def _generate_los(self, age: int, is_inflow: bool) -> int:
        """Generate length of service based on age"""
        max_possible_los = min(30, age - 18)

        if is_inflow:
            # New joiners have low LoS
            return self.rng.integers(0, min(5, max_possible_los + 1))

        # Sample from bands
        bands = self.config['length_of_service']['bands']
        band_options = list(bands.keys())
        band_probs = list(bands.values())

        band_idx = self.rng.choice(len(band_options), p=band_probs)
        band = band_options[band_idx]

        if band == "0-4":
            los_min, los_max = 0, 4
        elif band == "5-12":
            los_min, los_max = 5, 12
        else:  # 13-25
            los_min, los_max = 13, 25

        # Ensure valid range
        los_max = min(los_max, max_possible_los)
        if los_max < los_min:
            los_min = 0
            los_max = min(4, max_possible_los)

        if los_max >= los_min:
            return self.rng.integers(los_min, los_max + 1)
        else:
            return 0

    def _simulate_month(self, month: int) -> List[Dict]:
        """Simulate one month for all active personnel"""
        month_records = []
        outflows = []

        for person_id, state in self.active_personnel.items():
            person = state['master']

            # Update age
            state['age'] += 1/12
            state['los'] += 1/12
            state['survival_time'] += 1

            # JMES transition
            jmes_event = False
            old_jmes = state['jmes_current']
            new_jmes = self._transition_jmes(old_jmes)
            if new_jmes != old_jmes:
                jmes_event = True
                state['jmes_current'] = new_jmes

            # Injury event
            injury_type, injury_score = self._generate_injury(
                person.trade,
                state.get('deployment_status', DeploymentStatus.NOT_DEPLOYED),
                state.get('training_phase', TrainingPhase.NONE)
            )

            # Deployment
            deployment = self._generate_deployment(person.service_branch)
            state['deployment_status'] = deployment

            # Training
            training = self._generate_training()
            state['training_phase'] = training

            # Pregnancy (females only)
            pregnancy = PregnancyStatus.NOT_PREGNANT
            if person.gender == Gender.FEMALE:
                pregnancy = self._update_pregnancy(state)

            # Sick days
            sick_days = self._generate_sick_days(new_jmes, injury_type)

            # Outflow check
            outflow, reason = self._check_outflow(state, person)

            # Create person-month record
            pm = PersonMonth(
                person_id=person_id,
                month=month,
                age=state['age'],
                jmes_current=new_jmes,
                jmes_event_this_month=jmes_event,
                injury_type=injury_type,
                injury_severity_score=injury_score,
                deployment_status=deployment,
                training_phase=training,
                pregnancy_status=pregnancy,
                sick_days=sick_days,
                outflow_flag=outflow,
                outflow_reason=reason if outflow else None,
                inflow_flag=False,
                survival_time=state['survival_time'],
                censor_flag=(not outflow)
            )

            month_records.append(pm.to_dict())

            if outflow:
                outflows.append(person_id)

        # Remove outflows
        for pid in outflows:
            del self.active_personnel[pid]

        # Add inflows to maintain population
        n_inflows = len(outflows)
        for _ in range(n_inflows):
            new_person = self._create_person(is_inflow=True)
            self.all_personnel.append(new_person)

            self.active_personnel[new_person.person_id] = {
                'master': new_person,
                'jmes_current': new_person.baseline_jmes,
                'age': float(new_person.age_start),
                'los': new_person.length_of_service_start,
                'pregnancy_counter': 0,
                'survival_time': 0.0
            }

            # Create inflow record
            pm_inflow = PersonMonth(
                person_id=new_person.person_id,
                month=month,
                age=float(new_person.age_start),
                jmes_current=new_person.baseline_jmes,
                inflow_flag=True,
                survival_time=0.0,
                censor_flag=True
            )
            month_records.append(pm_inflow.to_dict())

        return month_records

    def _transition_jmes(self, current: JMESStatus) -> JMESStatus:
        """Transition JMES status"""
        trans = self.config['jmes']['transitions']

        if current == JMESStatus.MFD:
            if self.rng.random() < trans['MFD_to_MLD']:
                return JMESStatus.MLD
            elif self.rng.random() < trans['MFD_to_MND']:
                return JMESStatus.MND
        elif current == JMESStatus.MLD:
            if self.rng.random() < trans['MLD_to_MND']:
                return JMESStatus.MND
            elif self.rng.random() < trans['MLD_to_MFD']:
                return JMESStatus.MFD
        elif current == JMESStatus.MND:
            if self.rng.random() < trans['MND_to_MLD']:
                return JMESStatus.MLD
            elif self.rng.random() < trans['MND_to_MFD']:
                return JMESStatus.MFD

        return current

    def _generate_injury(self, trade: Trade, deployment: DeploymentStatus,
                        training: TrainingPhase) -> Tuple[InjuryType, int]:
        """Generate injury event"""
        base_rate = self.config['injuries']['baseline_monthly_mski']
        trade_mult = self.config['injuries']['trade_multipliers'][trade.value]

        # Adjust for deployment/training
        final_rate = base_rate * trade_mult
        if deployment != DeploymentStatus.NOT_DEPLOYED:
            final_rate *= self.config['injuries']['deployment_multiplier']
        if training == TrainingPhase.HIGH_RISK:
            final_rate *= self.config['injuries']['high_risk_training_multiplier']

        if self.rng.random() < final_rate:
            # Select injury type
            inj_types = self.config['injuries']['types']
            type_options = [InjuryType.MSKI_MINOR, InjuryType.MSKI_MAJOR,
                          InjuryType.MH_EPISODE, InjuryType.OTHER]
            type_probs = [inj_types['MSKI-minor'], inj_types['MSKI-major'],
                         inj_types['MH-episode'], inj_types['Other']]

            type_idx = self.rng.choice(len(type_options), p=type_probs)
            injury = type_options[type_idx]

            # Severity score
            if injury == InjuryType.MSKI_MINOR:
                score = self.rng.integers(1, 4)
            elif injury == InjuryType.MSKI_MAJOR:
                score = self.rng.integers(5, 10)
            else:
                score = self.rng.integers(3, 8)

            return injury, score

        return InjuryType.NONE, 0

    def _generate_deployment(self, service: ServiceBranch) -> DeploymentStatus:
        """Generate deployment status"""
        base_rate = self.config['deployment']['baseline_monthly_rate']
        service_mult = self.config['deployment']['service_rates'][service.value]

        final_rate = base_rate * service_mult

        if self.rng.random() < final_rate:
            # Select intensity
            intensities = self.config['deployment']['intensity']
            int_options = [DeploymentStatus.LOW_TEMPO, DeploymentStatus.HIGH_TEMPO]
            int_probs = [intensities['Low_tempo'], intensities['High_tempo']]

            int_idx = self.rng.choice(len(int_options), p=int_probs)
            return int_options[int_idx]

        return DeploymentStatus.NOT_DEPLOYED

    def _generate_training(self) -> TrainingPhase:
        """Generate training phase"""
        if self.rng.random() < self.config['training']['monthly_rate']:
            intensities = self.config['training']['intensity']
            int_options = [TrainingPhase.LOW_RISK, TrainingPhase.HIGH_RISK]
            int_probs = [intensities['Low_risk'], intensities['High_risk']]

            int_idx = self.rng.choice(len(int_options), p=int_probs)
            return int_options[int_idx]

        return TrainingPhase.NONE

    def _update_pregnancy(self, state: Dict) -> PregnancyStatus:
        """Update pregnancy status for females"""
        counter = state.get('pregnancy_counter', 0)

        if counter > 0:
            # Currently pregnant/postpartum
            state['pregnancy_counter'] = counter - 1

            preg_months = self.config['pregnancy']['duration']['pregnancy_months']
            postpartum_months = self.config['pregnancy']['duration']['postpartum_months']
            total_duration = preg_months + postpartum_months

            months_elapsed = total_duration - counter

            if months_elapsed <= 3:
                return PregnancyStatus.T1
            elif months_elapsed <= 6:
                return PregnancyStatus.T2
            elif months_elapsed <= 9:
                return PregnancyStatus.T3
            else:
                return PregnancyStatus.POSTPARTUM
        else:
            # Check for new pregnancy
            monthly_rate = self.config['pregnancy']['monthly_conception_rate']
            if self.rng.random() < monthly_rate:
                preg_months = self.config['pregnancy']['duration']['pregnancy_months']
                postpartum_months = self.config['pregnancy']['duration']['postpartum_months']
                state['pregnancy_counter'] = preg_months + postpartum_months
                return PregnancyStatus.T1

        return PregnancyStatus.NOT_PREGNANT

    def _generate_sick_days(self, jmes: JMESStatus, injury: InjuryType) -> int:
        """Generate sick days"""
        if injury != InjuryType.NONE:
            if injury == InjuryType.MSKI_MAJOR:
                return self.rng.integers(5, 15)
            else:
                return self.rng.integers(1, 7)
        elif jmes == JMESStatus.MND:
            return self.rng.integers(3, 10)
        elif jmes == JMESStatus.MLD:
            return self.rng.integers(0, 5)

        return 0

    def _check_outflow(self, state: Dict, person: PersonnelMaster) -> Tuple[bool, Optional[OutflowReason]]:
        """Check if person leaves service"""
        los = state['los']

        # Base monthly turnover rate
        monthly_rate = self.config['turnover']['monthly_rate']

        # LoS weighting
        los_weights = self.config['turnover']['los_weights']
        if los < 4:
            weight = los_weights['0-4']
        elif los < 12:
            weight = los_weights['5-12']
        else:
            weight = los_weights['13+']

        adjusted_rate = monthly_rate * weight

        if self.rng.random() < adjusted_rate:
            # Select reason
            if los < 4:
                return True, OutflowReason.PREMATURE_VOL
            elif los >= 20:
                return True, OutflowReason.NORMAL_TERM
            elif state['jmes_current'] == JMESStatus.MND:
                return True, OutflowReason.MEDICAL_DISCHARGE
            else:
                return True, OutflowReason.END_ENGAGEMENT

        return False, None

    def _verify_initial_cohort(self):
        """Verify initial cohort distributions"""
        print("\n=== Chain-of-Verification: Initial Cohort ===")

        master_list = [s['master'] for s in self.active_personnel.values()]

        # Service mix
        for service in ServiceBranch:
            actual = sum(1 for p in master_list if p.service_branch == service) / len(master_list)
            expected = self.config['service_mix'][service.value]
            print(f"  {service.value}: {actual:.1%} (expected {expected:.1%})")

        # Gender
        female_rate = sum(1 for p in master_list if p.gender == Gender.FEMALE) / len(master_list)
        print(f"  Female rate: {female_rate:.1%}")

        # Age
        ages = [p.age_start for p in master_list]
        print(f"  Mean age: {np.mean(ages):.1f}")

        # JMES
        mfd_rate = sum(1 for p in master_list if p.baseline_jmes == JMESStatus.MFD) / len(master_list)
        print(f"  MFD rate: {mfd_rate:.1%}")

        print("=" * 60)

    def _final_verification(self, master_df: pd.DataFrame, monthly_df: pd.DataFrame):
        """Final verification of simulation results"""
        print("\n" + "=" * 60)
        print("FINAL CHAIN-OF-VERIFICATION")
        print("=" * 60)

        # Population stability
        initial_pop = self.config['population']['initial_size']
        final_pop = len(self.active_personnel)
        stability = abs(final_pop - initial_pop) / initial_pop
        print(f"Population stability: {stability:.1%} (target < {self.config['validation']['population_stability']:.1%})")

        # Turnover calculation
        n_months = self.config['population']['simulation_months']
        total_outflows = monthly_df['outflow_flag'].sum()
        implied_annual_turnover = (total_outflows / initial_pop) / (n_months / 12)
        print(f"Implied annual turnover: {implied_annual_turnover:.1%} (target ~10%)")

        # JMES events
        jmes_events = monthly_df['jmes_event_this_month'].sum()
        print(f"JMES deterioration events: {jmes_events}")

        # Injuries
        total_injuries = (monthly_df['injury_type'] != 'None').sum()
        print(f"Total injuries: {total_injuries}")

        # Pregnancies
        pregnancies = (monthly_df['pregnancy_status'] != 'Not_pregnant').sum()
        print(f"Total pregnancies: {pregnancies}")

        print("=" * 60)

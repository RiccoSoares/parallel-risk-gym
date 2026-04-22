class CombatResolver:
    """Handles combat resolution with percentage-based casualties"""

    @staticmethod
    def resolve(attacking_troops, defending_troops):
        """Resolve combat deterministically

        Rules:
        - Defenders lose 70% of attacking troops (rounded down)
        - Attackers lose 60% of defending troops (rounded down)
        - If defenders reduced to <= 0: attacker wins and captures territory
        - Otherwise: defender holds with remaining troops

        Args:
            attacking_troops: Number of troops in the attack
            defending_troops: Number of troops defending

        Returns:
            tuple: (result, surviving_troops)
                result: 'attacker_wins' or 'defender_holds'
                surviving_troops: Troops remaining after combat
        """
        # Calculate casualties
        defender_casualties = int(attacking_troops * 0.7)
        attacker_casualties = int(defending_troops * 0.6)

        # Apply casualties
        defenders_remaining = defending_troops - defender_casualties
        attackers_remaining = attacking_troops - attacker_casualties

        if defenders_remaining <= 0:
            # Attacker wins - captures territory with remaining troops
            surviving_troops = max(1, attackers_remaining)
            return 'attacker_wins', surviving_troops
        else:
            # Defender holds
            return 'defender_holds', defenders_remaining

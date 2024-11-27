from vgc.behaviour import TeamBuildPolicy
from vgc.datatypes.Objects import PkmFullTeam, PkmRoster
from vgc.balance.meta import MetaData
from vgc.ecosystem.ChampionshipEcosystem import RandomTeamFromRoster
import mcts as mt


class MCTSRandomTeamBuilder(TeamBuildPolicy):
        
    def set_roster(self, roster: PkmRoster, ver: int = 0):
        self.roster = roster

    def get_action(self, s: MetaData) -> PkmFullTeam:
        mcts = None
        enemy_teams = self.__create_enemy_teams(self.roster, 20)
        # print([enemy.__str__() for enemy in enemy_teams])
        for enemy in enemy_teams:
            mcts = mt.train(mcts, self.roster, enemy)
            # print(mcts.get_team())
        return mcts.get_team()

    def __create_enemy_teams(self, roster, n_teams):
        teams_list = []
        for _ in range(n_teams):
            team = RandomTeamFromRoster(roster, 3, 4).get_team()
            teams_list.append(team)
        return teams_list

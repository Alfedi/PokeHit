from ExampleCompetitor import ExampleCompetitor
from vgc.balance.meta import StandardMetaData
from vgc.competition.Competitor import CompetitorManager
from vgc.ecosystem.ChampionshipEcosystem import ChampionshipEcosystem
from vgc.util.generator.PkmRosterGenerators import RandomPkmRosterGenerator
from MCTSCompetitor import MCTSCompetitor

N_OPPONENTS = 20  # 15 - 20


def main():
    generator = RandomPkmRosterGenerator(None, n_moves_pkm=4, roster_size=51)
    roster = generator.gen_roster()
    move_roster = generator.base_move_roster
    meta_data = StandardMetaData()
    meta_data.set_moves_and_pkm(roster, move_roster)
    ce = ChampionshipEcosystem(roster, meta_data, debug=False)

    mtcs_competitor = CompetitorManager(MCTSCompetitor("MCTS"))
    ce.register(mtcs_competitor)

    for i in range(N_OPPONENTS):
        cm = CompetitorManager(ExampleCompetitor("Enemy %d" % i))
        ce.register(cm)

    ce.run(n_epochs=10, n_league_epochs=10)

    dict = {competitor.competitor.name: competitor.elo for competitor in ce.league.competitors}
    file = open("clasificacion.txt", "a")
    file.write(str(sorted(dict.items(), key=lambda x: x[1], reverse=True)) + " \n")
    file.close()

    # print([competitor.competitor.name for competitor in ce.league.competitors ])
    # print([competitor.elo for competitor in ce.league.competitors ])
    print(ce.strongest().competitor.name)
    # print(ce.strongest().elo)


if __name__ == "__main__":
    main()

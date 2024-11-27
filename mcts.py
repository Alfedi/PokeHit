import numpy as np
import random

from node import *
import numpy as np
import data
from vgc.competition.Competitor import RandomPlayer
from vgc.competition.BattleMatch import PkmBattleEnv
from vgc.util.generator.PkmTeamGenerators import RandomTeamFromRoster


def create_team(roster):
    return RandomTeamFromRoster(roster, 3, 4).get_team()


def play(gamestate_, team, enemy_team):
    res = []
    gamestate = gamestate_.copy()
    gamestate["Team"] = team.__str__()
    agent, enemy_agent = RandomPlayer(), RandomPlayer()
    env = PkmBattleEnv((team.get_battle_team([0, 1, 2]), enemy_team.get_battle_team([0, 1, 2])), encode=(agent.requires_encode(), enemy_agent.requires_encode()), debug=True)
    n_battles = 3
    t = False
    battle = 0
    while battle < n_battles:
        log = ''
        s, _ = env.reset()
        while not t:
            a = [agent.get_action(s[0]), enemy_agent.get_action(s[1])]
            s, _, t, _, _ = env.step(a)
            log = env.log
            gamestate["Log"] += log 
        t = False
        battle += 1

        if (log.find("Trainer 0 Won") != -1):
            gamestate["Veredict"] += 1
            res.append(list((gamestate, True)))
        else:
            gamestate["Veredict"] -= 1
            res.append(list((gamestate, False)))
    return res


def train(mcts=None, roster=None, enemy=None):
    try:
        if mcts is None:
            mcts = Node(data.create_dict(roster), False, None, None)  # Gamestate, winner, team, parent

        node = mcts
        # Fase de selección
        while node.children is not None:
            ucts = [child.get_uct_score() for child in node.children]  # Best UCT score
            if None in ucts:
                node = random.choice(node.children)
            else:
                node = node.children[np.argmax(ucts)]

        # Fase de expansión
        new_team = create_team(roster)
        # Fase de simulación
        states = play(node.gamestate, new_team, enemy)  # Sacamos los gamestate de cada posible movimiento
        # print(states)
        node.set_children(
            [
                Node(state_winning[0], state_winning[1], team=new_team, parent=node)
                for state_winning in states
            ]  # Cada gamestate genera un hijo nuevo
        )
        # Comprobamos el mejor ganador en base al veredicto
        winner_nodes = [child for child in node.children if child.winner]

        # print(winner_nodes)
        if len(winner_nodes) > 0:
            best_winner = max(winner_nodes, key=lambda winner: winner["Veredict"])
            victory = best_winner
        else:
            victory = random.choice(node.children)

        # Fase de backpropagation
        parent = node
        while parent is not None:
            parent.n += 1
            if victory.winner:
                parent.w += 1
            parent = parent.parent
        return victory
    except Exception as error:
        print("TRAZA DE ERROR: ", error)

import numpy as np
import random
import requests
from vgc.behaviour import PkmFullTeam
from node import *
import numpy as np
import data
from vgc.competition.Competitor import RandomPlayer
from vgc.competition.BattleMatch import PkmBattleEnv
from vgc.util.generator.PkmTeamGenerators import RandomTeamFromRoster
import json
from vgc.datatypes.Constants import DEFAULT_PKM_N_MOVES
import sys, os
import re

def create_team(roster):
    return RandomTeamFromRoster(roster, 3, 4).get_team()


def play(gamestate_, team, enemy_team):
    res = []
    gamestate = gamestate_.copy()
    gamestate["Team"] = team.__str__()
    agent, enemy_agent = RandomPlayer(), RandomPlayer()
    env = PkmBattleEnv((team.get_battle_team([0, 1, 2]), enemy_team.get_battle_team([0, 1, 2])), encode=(agent.requires_encode(), enemy_agent.requires_encode()), debug=True)
    n_battles = 1
    t = False
    battle = 0
    while battle < n_battles:
        log = ''
        s, _ = env.reset()
        log = env.log
        gamestate["Log"] += log
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
            mcts = Node(data.create_dict([pkm.__str__() for pkm in roster]), False, None, None)  # Gamestate, winner, team, parent

        node = mcts
        # Fase de selección
        while node.children is not None:
            ucts = [child.get_uct_score() for child in node.children]  # Best UCT score
            if None is ucts:
                node = random.choice(node.children)
            else:
                node = node.children[np.argmax(ucts)]

        # Fase de expansión
        new_team = create_team(roster)
        # Fase de simulación
        states = play(node.gamestate, new_team, enemy)  # Sacamos los gamestate de cada posible movimiento
        # print(states)

        # file = open("log.txt", "w")
        # file.write(str(states))
        # file.close()

        node.set_children(
            [
                Node(state_winning[0], state_winning[1], team=new_team, parent=node)
                for state_winning in states
            ]  # Cada gamestate genera un hijo nuevo
        )

        # MODO RANDOM
        # Comprobamos el mejor ganador en base al veredicto
        # winner_nodes = [child for child in node.children if child.winner]

        # # print(winner_nodes)
        # if len(winner_nodes) > 0:
        #     best_winner = max(winner_nodes, key=lambda winner: winner["Veredict"])
        #     if (best_winner.gamestate["Veredict"] > node.gamestate["Veredict"]):
        #         victory = best_winner
        #     else:
        #         victory = node
        #     # victory = best_winner # Sin padre
        # else:
        #     victory = random.choice(node.children)
        ##########
        # MODO LLM
        headers = {'Content-Type': 'application/json'}
        payload = {
            'model': 'mistral-7b-instruct-v0.3',
            'messages': [
                {'role': 'user', 'content': 'You are a team building machine of a simplified version of pokémon. You will receive a Roster, a Team and a Veredict in a dictionary format. The Roster will be a list with the only possibilities to form a new team to win the next battle competition. The Team will be the last team used in the battle competition. And the Veredict will be a number that represents the number of victories in the last battle competition, the higher this number is the better. With this data you will have to make a new team from the Roster that will win the next battle competiton. Return the results in a list with the three indexes of the chosen pokémon from the Roster. JUST RETURN THE LIST OF THREE INDEXES. DO NOT WRITE ANYTHING ELSE. Return the response according with the following format inside the --- pattern: ---index1, index2, index3---, for example: ---14, 30, 50---. The response must be only simple text, no markdown. No explanation is needed. The teams must be of three pokémon, so just return 3 indexes in the specified format.'},
                {'role': 'user', 'content': str(states)},
            ],
            'max-tokens': -1
        }

        r = requests.post('http://192.168.1.23:1234/v1/chat/completions', data=json.dumps(payload), headers=headers)
        text_team = r.json()["choices"][0]["message"]["content"]
        # print(text_team)
        text_team = re.sub(r"</.*", '', text_team) # Nos quitamos la morralla
        text_team = text_team.replace('---', '') # Limpiamos el formato que le hemos especificado a la lista
        pkm_list = text_team.split(',')
        best_team = []
        # print(pkm_list)
        for i in [int(pkm) for pkm in pkm_list]:
            moves = random.sample(range(DEFAULT_PKM_N_MOVES), 4)
            if i >= len(roster) or i < 0:
                random_pkm = random.sample(range(len(roster)), 1)[0]
                best_team.append(roster[random_pkm].gen_pkm(moves)) # Si se está sacando un número que no existe pilla uno al azar y fuera
            else:
                best_team.append(roster[i].gen_pkm(moves))
        best_team = PkmFullTeam(best_team)
        # print(best_team)
        not_best_winner = Node(node.gamestate, False, best_team, node)
        node.set_children(node.children.append(not_best_winner))
        victory = not_best_winner
        # print(victory)

        # Fase de backpropagation
        parent = node
        while parent is not None:
            parent.n += 1
            if victory.winner:
                parent.w += 1
            parent = parent.parent
        return victory
    except Exception as error:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno, error)

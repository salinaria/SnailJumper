import copy
import numpy as np
from player import Player


class Evolution:
    def __init__(self):
        self.game_mode = 'Neuroevolution'

    def next_population_selection(self, players, num_players):
        '''
        Gets list of previous and current players (μ + λ) and returns num_players number of players based on their
        fitness value.

        :param players: list of players in the previous generation
        :param num_players: number of players that we return
        '''

        # TODO (Additional: Learning curve)
        self.learning_curve(players)

        return self.SUS(players, num_players)

    # Develop data for plotting
    def learning_curve(self, players):
        avg = 0
        maxi = players[0].fitness
        mini = players[0].fitness
        for player in players:
            avg += player.fitness
            if player.fitness > maxi:
                maxi = player.fitness
            if player.fitness < mini:
                mini = player.fitness
        avg /= len(players)
        data = f'{avg},{maxi},{mini}\n'
        with open('json.rj', 'a') as f:
            f.write(data)
        f.close()

    def generate_new_population(self, num_players, prev_players=None):
        '''
        Gets survivors and returns a list containing num_players number of children.

        :param num_players: Length of returning list
        :param prev_players: List of survivors
        :return: A list of children
        '''
        first_generation = prev_players is None
        if first_generation:
            return [Player(self.game_mode) for _ in range(num_players)]
        else:
            # TODO ( Parent selection and child generation )
            new_parents = [self.clone_player(player)
                           for player in prev_players]
            new_players = []

            parents = self.Q_tournament(new_parents, num_players)

            for i in range(0, num_players, 2):
                p1 = parents[i]
                p2 = parents[i + 1]
                c1, c2 = self.two_points_crossover(p1, p2)
                self.mutate(c1, 0.1)
                self.mutate(c2, 0.1)
                new_players.append(c1)
                new_players.append(c2)
            return new_players

    def clone_player(self, player):
        '''
        Gets a player as an input and produces a clone of that player.
        '''
        new_player = Player(self.game_mode)
        new_player.nn = copy.deepcopy(player.nn)
        new_player.fitness = player.fitness
        return new_player

    def top_k(self, players, num_players):
        players_top = sorted(players, key=lambda x: x.fitness, reverse=True)
        return players_top[: num_players]

    def roulette_wheel(self, players, num_players):
        population_fitness = sum([player.fitness for player in players])
        player_probabilities = [player.fitness /
                                population_fitness for player in players]
        return np.random.choice(players, num_players, p=player_probabilities)

    def SUS(self, players, num_players):
        population_fitness = sum([player.fitness for player in players])
        player_probabilities = []
        index = 0
        for player in players:
            player_probabilities.append(
                (player.fitness / population_fitness, index))
            index += 1
        player_probabilities.sort(key=lambda x: x[0], reverse=True)
        index = 0
        for i in range(len(player_probabilities)):
            player_probabilities[i] = (
                player_probabilities[i][0], player_probabilities[i][1], index, player_probabilities[i][0] + index)
            index += player_probabilities[i][0]
        choose_from = []
        sus_uniform = float(np.random.uniform(0, 1/num_players))
        j = 0
        for i in range(num_players):
            while j < len(player_probabilities) and not (player_probabilities[j][2] <= sus_uniform <= player_probabilities[j][3]):
                j += 1
            choose_from.append(players[player_probabilities[j][1]])
            sus_uniform += 1/num_players
        return choose_from

    def Q_tournament(self, players, num_players):
        Q = 15
        choosen_players = []
        for o in range(num_players):
            players_to_choose = np.random.choice(players, Q, replace=False)
            choosen_players.append(
                max(players_to_choose, key=lambda x: x.fitness))
        return choosen_players

    def mutate(self, player, threshold):
        weights = player.nn.weights
        biases = player.nn.biases

        for k in range(len(weights)):
            for i in range(weights[k].shape[0]):
                for j in range(weights[k].shape[1]):
                    if np.random.uniform(0, 1) < threshold:
                        weights[k][i, j] += np.random.normal(0, 1)
                if(np.random.uniform(0, 1) < threshold):
                    biases[k][i, 0] += np.random.normal(0, 1)

    def two_points_crossover(self, player1, player2):
        if(np.random.uniform(0, 1, 1) < 0.8):

            child1 = Player(self.game_mode)
            child1.nn = copy.deepcopy(player1.nn)
            weights_1 = child1.nn.weights
            biases_1 = child1.nn.biases

            child2 = Player(self.game_mode)
            child2.nn = copy.deepcopy(player2.nn)
            weights_2 = child2.nn.weights
            biases_2 = child2.nn.biases

            for i in range(len(player1.nn.weights)):

                shape_w, size_w = weights_1[i].shape, weights_1[i].size
                shape_b, size_b = biases_1[i].shape, biases_1[i].size

                weights_1[i].flatten()[size_w // 3: 2*size_w //
                                       3] = player2.nn.weights[i].flatten()[size_w // 3: 2*size_w // 3]
                weights_1[i].reshape(shape_w)
                biases_1[i].flatten()[size_b // 3: 2*size_b //
                                      3] = player2.nn.biases[i].flatten()[size_b // 3: 2*size_b // 3]
                biases_1[i].reshape(shape_b)

                weights_2[i].flatten()[size_w // 3: 2*size_w //
                                       3] = player1.nn.weights[i].flatten()[size_w // 3: 2*size_w // 3]
                weights_2[i].reshape(shape_w)
                biases_2[i].flatten()[size_b // 3: 2*size_b //
                                      3] = player1.nn.biases[i].flatten()[size_b // 3: 2*size_b // 3]
                biases_2[i].reshape(shape_b)

            return child1, child2
        else:
            return player1, player2

    def cal_crossover(self, player1, player2, alpha):
        if(np.random.uniform(0, 1, 1) < 0.8):

            child1 = Player(self.game_mode)
            child1.nn = copy.deepcopy(player1.nn)
            weights_1 = child1.nn.weights
            biases_1 = child1.nn.biases

            child2 = Player(self.game_mode)
            child2.nn = copy.deepcopy(player2.nn)
            weights_2 = child2.nn.weights
            biases_2 = child2.nn.biases

            for i in range(len(weights_1)):

                weights_1[i] = alpha * weights_1[i] + \
                    (1 - alpha) * weights_2[i]
                biases_1[i] = alpha * biases_1[i] + (1 - alpha) * biases_2[i]

                weights_2[i] = alpha * weights_2[i] + \
                    (1 - alpha) * weights_1[i]
                biases_2[i] = alpha * biases_2[i] + (1 - alpha) * biases_1[i]

            return child1, child2
        else:
            return player1, player2

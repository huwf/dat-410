import random
import copy



class RandomMoveMixin:
    def evaluate(self, possible_moves):
        return random.choice(tuple(possible_moves))

class MonteCarloMoveMixin:
    def evaluate(self, possible_moves):
        evaluations = {}

        for x in range(500):
            simulated_moves = []

            simulated_player = copy.deepcopy(self)

            simulated_other_player = copy.deepcopy(self)
            simulated_other_player.name = self.getNextPlayer(simulated_player)
            simulated_other_player.board = simulated_player.board

            current_player = simulated_player

            possible_moves = current_player.search(current_player.board)

            score = current_player.board.size * current_player.board.size

            while len(possible_moves) > 0:
                rnd = random.choice(tuple(possible_moves))

                current_player.place(rnd)

                #print(str(current_player.board))


                simulated_moves.append(rnd)

                if current_player.board.winner:
                    break

                score -= 1

                current_player = simulated_other_player if current_player.name == simulated_player.name else simulated_player

                possible_moves = current_player.search(current_player.board)


            first_move = simulated_moves[0]

            first_move_key = repr(first_move)

            if (not current_player.name == self.name) and current_player.board.winner:
                score *= -1

            if first_move_key in evaluations:
                evaluations[first_move_key] += score
            else:
                evaluations[first_move_key] = score

        most_optimal_move = []
        highest_score = 0

        for m,s in evaluations.items():
            if s > highest_score:
                highest_score = s
                most_optimal_move = eval(m)

        return most_optimal_move




    def getNextPlayer (self, current_player):

        if current_player.name == "X":
            return "O"
        
        return "X"



        
        

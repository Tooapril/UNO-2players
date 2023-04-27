
class UnoJudger:

    @staticmethod
    def judge_winner(payoffs):
        ''' Judge the winner of the game

        Args:
            players (list): The list of players who play the game

        Returns:
            (list): The player id of the winner
        '''
        if payoffs[0] > payoffs[1]:
            return [0]
        elif payoffs[0] < payoffs[1]:
            return [1]
        else:
            return None

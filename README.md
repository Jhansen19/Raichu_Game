#Raichu Game Report
By: Jonathan Hansen
Date: 10/2023

**The problem:**
Game of Raichu that contains various pieces that have different movements. The goal is to capture all the opponents’ pieces. The initial state: the initial state is the start of the board how it is given before any pieces move. The state space: this all spaces in the 2D board space. Successor function: this function gets all the potential moves for a given player from its current pieces’ locations. Goal State: the goal is when the opponent player has no pieces left on the board. Description:

- The program uses several functions to determine how to move, a couple these important ones are find_best_move, minimax, and make_move.
- The minimax algorithm uses iterative deepening and alpha beta pruning to lessen the search space.
- There were some issues at first with how the pieces were moving, namely this was because of changing the board type from string to 2D in different circumstances. After this was fixed I also had to ensure that wraparound movement was not allowed. So, we implemented a function to assure that pieces did not wrap around the board.
- I implemented a transpositional table to ensure that I did not revisit potential moves when searching for the best move.
- One of the biggest problems I had was getting each function to work together and assuring that the board was only searched in the 2D space. I believe this was particularly challenging for me because at first, I developed the program to work on the 1D string as it is in the main function.
- I added values to the pieces and advancement values so that Pichus and Pikachu’s gain value as they advance to the end of the board to become a Raichu.
- At the point of writing this I am trying to figure out how to get the pieces to move correctly. The functions creating the moves give back lists of lists that have the correct values of moves for each character. However, when that move is performed it is not the correct move. I am not currently sure how this is happening when the moves generated are in fact correct.

- 

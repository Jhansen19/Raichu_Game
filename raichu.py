# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 08:42:51 2023

@author: Jon
"""

#!/usr/local/bin/python3
#
# B551 Fall 2023
# Professor Saúl Blanco
# Do not share these assignments or their solutions outside of this class.
#
# raichu.py : Play the game of Raichu
#
# Submitted by : [Jonathan Hansen]
#

# =============================================================================
# . --> squares have no piece
# w --> white Pichu
# W --> white Pikachu
# Q --> white Raichu
# b --> black Pichu
# B --> black Pikachu
# $ --> black Raichu
# =============================================================================


import sys
import time
# import numpy


def board_to_string(board, N):
    return "\n".join(board[i:i + N] for i in range(0, len(board), N))

###################### MOVES CHECKS #################################################

def is_valid_square(x, y, N):
    return 0 <= x < N and 0 <= y < N


def is_empty(board, N, x, y):
    # Check if the coordinates are within the board
    if not is_valid_square(x, y, N):
        return False
    
    # Check the value at the specified coordinates in the 2D board
    if board[x][y] == '.':
        return True
    else:
        return False

def is_legal_move(start, end):
    # Check for side-wrap
    if start[1] == 0 and end[1] == N-1:
        return False
    if start[1] == N-1 and end[1] == 0:
        return False
    # Check for end-to-end
    if start[0] == 0 and end[0] == N-1:
        return False
    if start[0] == N-1 and end[0] == 0:
        return False
    return True

def is_opponent(board, N, x, y, player):
    #print("What is being passed to is_opponent??:", board, N, x, y, player)

    # Convert 2D coordinates (x, y) to 1D index
    index = x * N + y
    
    # Check if the index is within the bounds of the board string
    if index < 0 or index >= len(board):
        return False  # Index is out of bounds

    piece = board[index]
    if player == 'w':
        return piece in ['b', 'B', '$']
    elif player == 'b':
        return piece in ['w', 'W', 'Q']
    return False


def can_capture(board, N, player, piece_type, x, y, dx, dy):
    # The jump position
    jump_x, jump_y = x + dx, y + dy

    # Check if the jump position is within the board
    if not is_valid_square(jump_x, jump_y, N):
        return False

    opponent = board[jump_x][jump_y]
    
    # Rules for capturing based on piece types
    if piece_type == 'w':  # White Pichu
        if opponent == 'b':
            return True
    elif piece_type == 'b':  # Black Pichu
        if opponent == 'w':
            return True
    elif piece_type == 'W':  # White Pikachu
        if opponent in ['b', 'B']:
            return True
    elif piece_type == 'B':  # Black Pikachu
        if opponent in ['w', 'W']:
            return True
    elif piece_type == 'Q':  # White Raichu
        if opponent in ['b', 'B', '$']:
            return True
    elif piece_type == '$':  # Black Raichu
        if opponent in ['w', 'W', 'Q']:
            return True
    
    return False

###################### MOVES SECTION #################################################

def pichu_moves(board, N, position, player):
    #print(f"DEBUG: Starting pichu_moves with position: {position}, player: {player}")
    # Convert position if it's an integer
    
    if isinstance(position, int):
        x, y = divmod(position, N)
    else:
        x, y = position
    
    moves = []
    direction = 1 if player == 'w' else -1
    
    # Simple forward diagonal move
    left_diag = (x - 1, y + direction)
    right_diag = (x + 1, y + direction)

    if is_valid_square(left_diag[0], left_diag[1], N) and is_empty(board, N, left_diag[0], left_diag[1]):
        moves.append(((x, y), left_diag))

    if is_valid_square(right_diag[0], right_diag[1], N) and is_empty(board, N, right_diag[0], right_diag[1]):
        moves.append(((x, y), right_diag))

    # Jump over opponent's Pichu (modified to use can_capture)
    jump_left = (x - 2, y + 2*direction)
    jump_right = (x + 2, y + 2*direction)
    
    if is_valid_square(jump_left[0], jump_left[1], N) and is_empty(board, N, jump_left[0], jump_left[1]) and can_capture(board, N, player, board[x][y], x, y, -2, 2*direction):
        moves.append(((x, y), jump_left))
    
    if is_valid_square(jump_right[0], jump_right[1], N) and is_empty(board, N, jump_right[0], jump_right[1]) and can_capture(board, N, player, board[x][y], x, y, 2, 2*direction):
        moves.append(((x, y), jump_right))


    # Filter out illegal moves
    moves = [move for move in moves if is_legal_move((x, y), move[1])]
    
    # print("printing moves in pichu moves to see what we are returning:", moves)
    return moves


def get_pikachu_moves(board, N, position, player):
    if player == 'white':  # Assuming 'white' moves upwards
        directions = [(-1, 0), (0, 1), (0, -1)]  # Up, Right, Left
    else:  # 'black' moves downwards
        directions = [(1, 0), (0, 1), (0, -1)]  # Down, Right, Left

    moves = []
    
    # Rest of the function remains the same

    
    for dx, dy in directions:
        # Move 1 or 2 squares to an empty square, as long as all squares in between are also empty
        for distance in [1, 2]:
            new_x, new_y = position[0] + dx * distance, position[1] + dy * distance
            if is_valid_square(new_x, new_y, N) and is_empty(board, N, new_x, new_y):
                all_empty = True
                for step in range(1, distance):
                    if not is_empty(board, N, position[0] + dx * step, position[1] + dy * step):
                        all_empty = False
                        break
                if all_empty:
                    moves.append(((position[0], position[1]), (new_x, new_y)))

        # Check for jumps over 1 or 2 squares
        # Check for jumps over 1 or 2 squares
        for distance in [2, 3]:  # Jump over 1 or 2 squares
            new_x, new_y = position[0] + dx * distance, position[1] + dy * distance
            if is_valid_square(new_x, new_y, N) and is_empty(board, N, new_x, new_y):
                # Check if all squares between the start position and the square before the jump are empty
                all_empty = True
                for step in range(1, distance - 1):
                    if not is_empty(board, N, position[0] + dx * step, position[1] + dy * step):
                        all_empty = False
                        break
        
                # Check if there's an opponent piece to jump over
                if all_empty:
                    jump_over_x, jump_over_y = position[0] + dx * (distance - 1), position[1] + dy * (distance - 1)
                    if can_capture(board, N, player, board[position[0]][position[1]], jump_over_x, jump_over_y, dx, dy):
                        moves.append(((position[0], position[1]), (new_x, new_y)))

    
    legal_moves = []
    for move in moves:
        if is_legal_move(move[0], move[1]):
            legal_moves.append(move)
    
    return legal_moves


def get_raichu_moves(board, N, position, player):
    directions = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]  # 8 possible directions
    moves = []

    for dx, dy in directions:
        distance = 1
        jumped_over_piece = False
        while True:
            new_x, new_y = position[0] + dx * distance, position[1] + dy * distance
            if is_valid_square(new_x, new_y, N):
                if is_empty(board, N, new_x, new_y):
                    moves.append(((position[0], position[1]), (new_x, new_y)))
                    distance += 1
                elif can_capture(board, N, player, board[position[0]][position[1]], new_x, new_y, dx, dy) and not jumped_over_piece:
                    # Can jump over one opponent piece
                    jumped_over_piece = True
                    moves.append(((position[0], position[1]), (new_x, new_y)))
                    distance += 1
                elif is_opponent(board, N, new_x, new_y, player):
                    # Encountered a second opponent's piece, cannot jump two pieces
                    break
                else:
                    # Encountered its own piece, stop checking further in this direction
                    break
            else:
                break
    
    legal_moves = []
    for move in moves:
        if is_legal_move(move[0], move[1]):
            legal_moves.append(move)
    
    return legal_moves

def possible_moves(board_string, player):
    """
    Generate all possible moves for the given player on the provided board.
    """
    moves = []
    # Convert the string representation of the board to a 2D list
    board = [list(row) for row in board_string.strip().split('\n')]
    N = len(board[0])  # Assuming all rows have the same length
    #print("printing board in possible moves:", board)    
    #print("printing N in possible moves:", N)
    # Iterate over the board to get positions of the player's pieces
    for row in range(len(board)):
        for col in range(len(board[row])):
            piece = board[row][col]

            # Check if the piece belongs to the current player
            if piece.lower() == player and piece.isalpha():
                
                # Pichu
                if piece in 'wb':
                    moves.extend(pichu_moves(board, N, (row, col), player))
                
                # Pikachu
                elif piece in 'WB':  
                    moves.extend(get_pikachu_moves(board, N, (row, col), player))

                # Raichu
                elif piece in 'Q$':
                    moves.extend(get_raichu_moves(board, N, (row, col), player))
    
    
    # DEBUG: Print current board state and possible moves
    #print("DEBUG Current Board:")
    #print(board_to_string(board, N))
    # print("DEBUG Possible Moves for player", player, ":", moves)
    return moves

##################### GAME BOARD, EVALUATION, AND MINIMAX ########################

def calculate_piece_value(piece, rank):
    base_values = {'w': 2, 'W': 4, 'Q': 9, 'b': -2, 'B': -4, '$': -9}
    advancement_bonus = 1
    
    if piece in ['w', 'W', 'b', 'B']:
        if piece.islower():  # 'w' or 'b'
            advancement = 8 - rank
        else:  # 'W' or 'B'
            advancement = rank - 1
        
        return base_values[piece] + advancement * advancement_bonus
    
    return base_values[piece]

def check_promotion(piece, rank):
    # print(f"Checking promotion for piece '{piece}' at rank {rank}")  # Debug print
    if piece in ['w', 'W'] and rank == 8:
        return 'Q'  # Promote white pieces to 'Q' when they reach rank 8
    elif piece in ['b', 'B'] and rank == 1:
        return '$'  # Promote black pieces to '$' when they reach rank 1
    return piece


CAPTURE_BONUS = 10  # You can adjust this value as per your game's strategy

def calculate_capture_score(captured_pieces, player):
    score = 0
    if player == 'w':
        # Calculate the score for all pieces captured by 'w' player
        for piece, count in captured_pieces.items():
            if piece in ['b', 'B', '$']:  # Pieces that 'w' player can capture
                score += count * CAPTURE_BONUS
    else:
        # Calculate the score for all pieces captured by 'b' player
        for piece, count in captured_pieces.items():
            if piece in ['w', 'W', 'Q']:  # Pieces that 'b' player can capture
                score -= count * CAPTURE_BONUS

    return score


def evaluate(board, player):
    score = 0
    for row in range(len(board)):
        for col in range(len(board[row])):
            piece = board[row][col]
            if piece in ["w", "W", "Q", "b", "B", "$"]:  # Valid pieces
                score += calculate_piece_value(piece, row)

                # Check for safety by being behind or diagonal to another piece
                if provides_support(board, row, col, piece):
                    score += .5  # Adjust the value as per your game's strategy
                
                # Check if a jump is possible
                if can_jump_opponent(board, row, col, piece):
                    score += .8  # bonus for being able to jump an opponent piece
            

    return score

def can_jump_opponent(board, row, col, piece):
    # Assuming 'player' is the player who owns the 'piece'
    player = 'w' if piece.islower() else 'b'
    
    # Directions in which a jump might be possible
    directions = [
        (0, 1), (1, 0), (0, -1), (-1, 0),  # Horizontal and vertical for Pikachu
        (1, 1), (-1, 1), (-1, -1), (1, -1)  # Diagonals for Raichu
    ]

    for dx, dy in directions:
        if can_capture(board, len(board), player, piece, row, col, dx, dy):
            # A jump is possible in this direction
            return True

    return False

def provides_support(board, row, col, piece):
    # Assuming 'player' is the player who owns the 'piece'
    player = 'w' if piece.islower() else 'b'
    
    # Determine the direction of movement based on the player
    direction = 1 if player == 'w' else -1

    # Check positions behind and diagonally behind the current piece
    support_positions = [
        (row - direction, col),         # Directly behind
        (row - direction, col - 1),     # Diagonally behind left
        (row - direction, col + 1),     # Diagonally behind right
    ]

    for r, c in support_positions:
        if is_valid_square(r, c, len(board)) and board[r][c].lower() == player:
            # Found a piece of the same player behind or diagonally behind
            return True

    return False


def make_move(board, start, end, N, player):
    if isinstance(board, str):
        board = [list(row) for row in board.strip().split('\n')]
    
    start_x, start_y = start
    end_x, end_y = end
    piece = board[start_x][start_y]  # Get the piece at the start position

    # Check if the starting position is empty or doesn't contain the player's piece
    if piece == '.' or ((player == 'w' and piece not in ['w', 'W', 'Q']) or (player == 'b' and piece not in ['b', 'B', '$'])):
        return board  # Skip the move and return the board unchanged

    promoted_piece = check_promotion(piece, end_x + 1)
    board[end_x][end_y] = promoted_piece
    board[start_x][start_y] = '.'

    # # Handle capturing in a separate function
    # handle_capture(board, start_x, start_y, end_x, end_y, piece)
    # print("Board after move:")
    # print(board)
    return board



# function for the core Minimax search
def minimax(board, depth, maximizing_player, alpha, beta, player):
    # print("Entering minimax at depth:", depth)
    # print("Board state:")
    # print(board)
    
    state_key = '\n'.join(''.join(row) for row in board)

    if state_key in transposition_table:
        # print("Found in transposition table:", state_key)
        return transposition_table[state_key]

    if depth == 0 or game_over(board, player):
        # print("Base case reached at depth:", depth)
        return evaluate(board, player)

    best_value = float('-inf') if maximizing_player else float('inf')
    board_string = '\n'.join(''.join(row) for row in board)
    moves = possible_moves(board_string, player)
    # print(f"Generated {len(moves)} possible moves.")

    for move in moves:
        # print(f"Exploring move: {move} for player {player} at depth: {depth}")
        new_board = make_move(board, move[0], move[1], len(board), player)
        evaluation = minimax(new_board, depth - 1, not maximizing_player, alpha, beta, 'b' if player == 'w' else 'w')

        
        # print(f"Evaluation for move {move}: {evaluation} for player {player} at depth: {depth}")
        
        if maximizing_player:
            best_value = max(best_value, evaluation)
            alpha = max(alpha, best_value)
        else:
            best_value = min(best_value, evaluation)
            beta = min(beta, best_value)

        if beta <= alpha:
            # print(f"Pruning at move: {move} for player {player}")
            break

    transposition_table[state_key] = best_value
    # print(f"Best value for depth {depth} for player {player}: {best_value}")
    return best_value


transposition_table = {}

def game_over(board, player):
    
    # If it's white's turn, check for the absence of black pieces
    if player == 'w':
        b_pieces = sum(1 for row in board for piece in row if piece in ["b", "B", "$"])
        return b_pieces == 0
    
    # If it's black's turn, check for the absence of white pieces
    elif player == 'b':
        w_pieces = sum(1 for row in board for piece in row if piece in ["w", "W", "Q"])
        return w_pieces == 0

def find_best_move(board, N, player, timelimit):
    board = board_to_string(board, N)
    start_time = time.time()
    depth = 6
    best_move = None
    best_move_from_last_iteration = None  # Store the best move from the last iteration
    maximizing_player = player == 'w'
    
    while time.time() - start_time < timelimit:
        alpha, beta = float('-inf'), float('inf')
        best_val = float('-inf') if maximizing_player else float('inf')

        possible_moves_list = possible_moves(board, player)
        # print(f"Evaluating {len(possible_moves_list)} possible moves at depth {depth}...")

        # Move ordering: evaluate the best move from the last iteration first
        if best_move_from_last_iteration:
            possible_moves_list.remove(best_move_from_last_iteration)
            possible_moves_list.insert(0, best_move_from_last_iteration)

        for move in possible_moves_list:
            new_board = make_move(board, move[0], move[1], N, player)
            move_val = minimax(new_board, depth, not maximizing_player, alpha, beta, player)
            
            if maximizing_player and move_val > best_val:
                best_val, best_move = move_val, move
                alpha = max(alpha, move_val)
                
            elif not maximizing_player and move_val < best_val:
                best_val, best_move = move_val, move
                beta = min(beta, move_val)
                
            if time.time() - start_time > timelimit:
                # print("Time limit exceeded, breaking out of loop.")
                break

        # Update the best move from this iteration to be used in the next iteration
        best_move_from_last_iteration = best_move
        depth += 1

    if best_move is not None:
        # print("printing items before final_board_state", board, best_move)
        final_board_state = make_move(board, best_move[0], best_move[1], N, player)
        final_board_string = ''.join(''.join(row) for row in final_board_state)
        yield final_board_string
    else:
        # print("No valid move found. Returning initial board state.")
        initial_board_string = '\n'.join(''.join(row) for row in board)
        return [initial_board_string]


def print_2d_board(board, N):
    board_2d = board_to_string(board, N)
    print(board_2d)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise Exception("Usage: Raichu.py N player board timelimit")
        
    (_, N, player, board, timelimit) = sys.argv
    N = int(N)
    timelimit = int(timelimit)
    if player not in "wb":
        raise Exception("Invalid player.")
    
    #print("test print board, player, and timelimit", _, N, player, board, timelimit)
    
    if len(board) != N * N or 0 in [c in "wb.WB@$" for c in board]:
        raise Exception("Bad board string.")

    print("Searching for best move for " + player + " from board state: \n" + board_to_string(board, N))
    print("Here's what I decided:")
    for new_board in find_best_move(board, N, player, timelimit):
        print(new_board)
        print(print_2d_board(new_board, N))
        






        
#RESOURCES
#https://chat.openai.com/
#
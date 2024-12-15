import chess
import chess.engine
import random

engine_path = r"stockfish\stockfish-windows-x86-64-avx2.exe"
engine = chess.engine.SimpleEngine.popen_uci(engine_path)

import chess

# def captured(board, move):
#     target_square = move.to_square
#     return board.piece_at(target_square) is not None

def rooks_are_connected(board):#after the move => board_after=>if student white the next turn will be black
    if board.turn: #the student play with black
        rooks = [square for square in chess.SQUARES if board.piece_at(square) and board.piece_at(square).symbol() == 'r']
    else:    
        rooks = [square for square in chess.SQUARES if board.piece_at(square) and board.piece_at(square).symbol() == 'R']

    if len(rooks) == 2:
        ranks = [chess.square_rank(rook) for rook in rooks]
        files = [chess.square_file(rook) for rook in rooks]


        if ranks[0] == ranks[1]:  # Same rank
            min_file, max_file = min(files), max(files)
            return all(board.piece_at(chess.square(file, ranks[0])) is None for file in range(min_file + 1, max_file))


    return False

def is_brilliant(board, move):

    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        eval_before = engine.analyse(board, chess.engine.Limit(depth=20))["score"].relative
        

        analysis = engine.analyse(board, chess.engine.Limit(depth=20))
        optimal_move = analysis["pv"][0]

        # Check if the move is optimal
        if move == optimal_move:
            # Simulate the move
            board.push(move)
            
            # Evaluate the position after the move
            eval_after = engine.analyse(board, chess.engine.Limit(depth=20))["score"].relative
            board.pop()

            # Check for a sacrifice (losing material)
            piece_captured = board.is_capture(move)
            piece_lost = board.piece_at(move.from_square)

            if piece_captured or piece_lost.piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                # Ensure the move leads to a significant advantage
                if eval_after > eval_before:
                    return "You played a brilliant move! Sacrificing material for a winning position. Good job!"

        return "Your move was good but not classified as a brilliant sacrifice."




def get_best_move(board, time_limit: float = 2.0) -> str:

    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        limit = chess.engine.Limit(time=time_limit)
        
        result = engine.play(board, limit)
        best_move = result.move
    
    return best_move 


def detect_forks_and_pins(board,move):

    to_square = move.to_square
    piece = board.piece_at(to_square)


    if piece.piece_type not in [chess.ROOK, chess.BISHOP, chess.QUEEN]:
        return "none"

    # Define directional offsets for each piece type
    directions = {
        chess.ROOK: [(0, 1), (1, 0), (0, -1), (-1, 0)],  # Horizontal & vertical
        chess.BISHOP: [(1, 1), (1, -1), (-1, 1), (-1, -1)],  # Diagonals
        chess.QUEEN: [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)],  # Both
    }

    board_size = 8  # Chessboard is 8x8

    # Convert square index to file and rank
    from_file, from_rank = chess.square_file(to_square), chess.square_rank(to_square)

    for offset_file, offset_rank in directions[piece.piece_type]:
        pinned_piece = None
        blocking_piece = None

        file, rank = from_file + offset_file, from_rank + offset_rank

        while 0 <= file < board_size and 0 <= rank < board_size:
            current_square = chess.square(file, rank)
            target = board.piece_at(current_square)

            if target:
                if target.color == piece.color:
                    break  # Friendly piece blocks the pin
                elif pinned_piece is None:
                    pinned_piece = current_square  # First enemy piece might be pinned
                else:
                    blocking_piece = current_square
                    break  # Second enemy piece blocks the pin
            file += offset_file
            rank += offset_rank

        # If the pinned piece shields the king, it's a valid pin
        if pinned_piece and blocking_piece:
            pinned_piece_obj = board.piece_at(pinned_piece)
            blocking_piece_obj = board.piece_at(blocking_piece)
            if blocking_piece_obj and blocking_piece_obj.piece_type == chess.KING:
                return f"Your {piece.symbol()} has pinned the {pinned_piece_obj.symbol()} to the King."

    # Check if the piece attacks two or more opponent pieces------------FORK--------------------------------------------------------------------
    attacked_squares = board.attacks(to_square)
    attacked_pieces = [
        target for target in attacked_squares
        if board.piece_at(target) and board.piece_at(target).color != piece.color
    ]
    # Filter out only the valuable pieces
    valuable_pieces = [target for target in attacked_pieces if board.piece_at(target).piece_type in [chess.ROOK,chess.KNIGHT,chess.BISHOP, chess.QUEEN, chess.KING]]

    if len(valuable_pieces) < 2:
        return "none"

    # Ensure the piece cannot be captured
    if board.is_attacked_by(not piece.color, to_square):#could be captured
        return "none"

    # Generate explanation
    attacked_pieces_names = [str(board.piece_at(target)) for target in valuable_pieces]
    return f"Your {piece.symbol()} forked the {' and '.join(attacked_pieces_names)}, and it cannot be captured."


                                            #0          1    2          3       4                5
def explain_move(fen_before, fen_after, move,evaluate):
    # Initialize boards
    explanation=""
    board_before = chess.Board(fen_before)
    board_after = chess.Board(fen_after)

    initial_square = move.from_square
    initial_piece = board_before.piece_at(move.from_square)

    target_square = move.to_square 
    target_piece = board_after.piece_at(target_square)

    try:
        # Analyze the before and after positions
        analyse_before = engine.analyse(board_before, chess.engine.Limit(time=0.1))
        analyse_after = engine.analyse(board_after, chess.engine.Limit(time=0.1))
        #print(analyse_after.get("pv", [])) best next moves 

        # Get the best move
        best_move_to_play = get_best_move(board_before)
       

        print("you played:", move)

        before_eval = analyse_before['score'].relative.score()/100 if analyse_before['score'].relative.score() is not None else 0
        after_eval = analyse_after['score'].relative.score()/100 if analyse_after['score'].relative.score() is not None else 0

        print(f"Before evaluation: {before_eval}")
        print(f"After evaluation: {after_eval}")

        difference=after_eval-before_eval

        # if difference>0 and board_before.turn or difference<0 and not board_before.turn  :
        #     explanation="improvmenet"
        # else :
        #     explanation="opposit"

        if not analyse_after['score'].is_mate():
            

            if board_before.fullmove_number <6 :
                # if board_before.turn:
                if chess.square_name(initial_square)in["b2","d2","e2","g2","b7","d7","e7","g7"]:
                    explanation="It is nice to open the diagonals for your bishop"#this move opens the path for your bishop to enter the game 
                if initial_piece.piece_type  in [chess.KNIGHT, chess.BISHOP]    :
                    explanation="you followed a great chess principle and developed a new piece  "#knight is tricky isnt  
                if  initial_piece.piece_type==chess.KNIGHT:
                    explanation="you develop a knight which means you move it from its starting position"   
                if initial_piece==chess.BISHOP:
                    explanation="you further develop your pieces by moving the bishop"  
                if initial_piece==chess.QUEEN:
                    explanation="its good to get your queen off of its starting square and into the action"

            if rooks_are_connected(board_after):
                explanation="Your rooks are connected on the same rank, which helps them work together."

            if move==best_move_to_play and board_after.is_capture(get_best_move(board_after)) and  initial_piece in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT] :
                explanation="You played a brilliant move! Sacrificing material for a winning position. Good job!"
                evaluate[2]+=1
            #---------------------------------------------------------------------------------------------------------------
            
            #---------------------------------------------------------------------------------------------------------------
            if move==best_move_to_play:
                explanation="wow you played the optimal move from the engine great job"    
                evaluate[5]+=1
            #---------------------------------------------------------------------------------------------------------------
            if detect_forks_and_pins(board_after,move)!="none":
                explanation=detect_forks_and_pins(board_after,move)
                evaluate[4]+=1

            # if board_before.is_capture(best_move_to_play) and target_piece in best_move_to_play.target_piece:
            #     if not board_before.is_capture(move) or target_piece not in move.target_piece:
            #         explanation = "You missed the opportunity to capture " + target_piece 
            #         miss+=1   
            #---------------------------------------------------------------------------------------------------------------
            # else:

            #     print(f"Evaluation difference: {difference}")
            #     if 300 <= difference < 500:
            #         explanation = "Great"
            #     elif 100 <= difference < 300:
            #         explanation = "Best move!"
            #     elif -50 < difference < 100:
            #         explanation = "Good move."
            #     # elif -100 <= difference <= -50:
            #     #     explanation = "Miss"
            #     elif -300 <= difference < -50:#-100
            #         explanation = "you did a mistake "
            #         evaluate[0]+=1
            #     else:  # difference < -300
            #         explanation = "you lost position"
            #         evaluate[3]+=1
        else:
            explanation="there is a mate"            
    except (chess.engine.EngineTerminatedError, chess.engine.EngineError, KeyError) as e:
        explanation = f"Unable to evaluate the move quality due to engine limitations. Error: {e}"
    finally:
        pass  # Ensure the engine is always closed

    return explanation

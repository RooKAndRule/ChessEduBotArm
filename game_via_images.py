from board_detction import get_squares
from get_position import get_fen_from_board
from GUIPlayer import show_board,choose_color,show_id_manager_interface,ask_to_play_another,setup_realtime_board_view
import matplotlib.pyplot as plt
from comments import explain_move
# from code_Arm import move_piece,cleanup,get_pict
import tensorflow as tf 
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import pickle
import torch
from torchvision import transforms
from PIL import Image
import cv2
import chess
import chess.engine
import chess.pgn
import os
import numpy as np
import datetime
import chess
from tensorflow.keras.models import load_model



#-------------------------------------------------------------------------------------------------------------------------------------------   


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COUNTER_FILE = "game_counter.txt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

engine_path=r"stockfish\stockfish-windows-x86-64-avx2.exe"
engine = chess.engine.SimpleEngine.popen_uci(engine_path)

#loading models--------------------------------------------------------------------------------------------------------------------------

model_for_piece = torch.load('models\chess_piece_classifier.pth')  
model_for_piece = model_for_piece.to(DEVICE)  

# Load the color model
model_for_color = torch.load('models\chess_color_classifier.pth') 
model_for_color = model_for_color.to(DEVICE)  



#---functions deal with changes on the board----------------------------------------------------------------------------------------------------------------------------------------    

def get_moved_piece(predicted_move, board):
    # Start Stockfish engine
    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        # Make the predicted move on the board
        move = chess.Move.from_uci(predicted_move)
        board.push(move)

        # Get the piece that moved
        moved_piece = board.piece_at(move.from_square)

        # Map the piece to a human-readable form
        piece_map = {
            chess.PAWN: "Pawn",
            chess.KNIGHT: "Knight",
            chess.BISHOP: "Bishop",
            chess.ROOK: "Rook",
            chess.QUEEN: "Queen",
            chess.KING: "King"
        }
        
        # Get the piece type that was moved
        moved_piece_type = piece_map.get(moved_piece.piece_type, "Unknown Piece")
        
        return moved_piece_type
    



def get_move(fen, level="beginner"):
    board = chess.Board(fen)
    
    # Set options based on player level
    if level == "beginner":
        options = {
            "Skill Level": 5,          
            "UCI_LimitStrength": True, # Limit engine strength
            "UCI_Elo": 1350           
        }
        time_limit = 0.1  
    elif level == "pro":
        options = {
            "Skill Level": 15,         
            "UCI_LimitStrength": False # Full engine strength
        }
        time_limit = 0.5  

    # Apply the options to Stockfish
    for option, value in options.items():
        engine.configure({option: value})

    # Get the best move within the time limit
    result = engine.play(board, chess.engine.Limit(time=time_limit))
    return result.move



def get_move_from_fen(fen_before, fen_after):
    board_before = chess.Board(fen_before)
    board_after = chess.Board(fen_after)
    
    legal_moves = list(board_before.legal_moves)

    for move in legal_moves:
        board_before.push(move)

        if board_before.board_fen() == board_after.board_fen():
            return move

        board_before.pop()
        
    
    return None  
#---functions deal with changes on the board----------------------------------------------------------------------------------------------------------------------------------------    
   
  

class ChessGame:
    def __init__(self,level,id,name):
        #,mistakes,miss,brilliants,blunders,forks_and_pins,optimal_moves
        self.board = chess.Board()
        self.active_color = 'White'
        self.playerid=id
        self.playername=name
        self.playerlevel=level
        self.moves = []
        self.result = None
        self.student_color =choose_color()
        self.arm_color =list({"White","Black"}-{self.student_color})[0]
        self.mistakes=0
        self.miss=0
        self.brilliants=0
        self.blunders=0
        self.forks_and_pins=0
        self.optimal_moves=0

        self.middle_game_start = False  # Flag for middle game start

    def update(self, values):
        if len(values) != 6:
            print("Error: The input list must contain exactly 6 values.")
            return
        
        self.mistakes += values[0]
        self.miss += values[1]
        self.brilliants += values[2]
        self.blunders += values[3]
        self.forks_and_pins += values[4]
        self.optimal_moves += values[5]    

    def make_move(self, move):
        if isinstance(move, chess.Move) and move in self.board.legal_moves:
            self.board.push(move)
            self.moves.append(move)   
            return True
        else:
            print(self.board.legal_moves)
            print("Illegal move!")
            return False


    def save_to_pgn(self, filename="game.pgn"):

        
        game = chess.pgn.Game()
        game.headers["Event"] = "TSPY12"
        game.headers["Site"] = "local"
        game.headers["Date"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        game.headers["White"] = self.playername if self.student_color=="White" else "arm" #id_p
        game.headers["Black"] = self.playername if self.student_color=="Black" else "arm"
        game.headers["id"]=self.playerid
        game.headers["level"]=self.playerlevel
        game.headers["mistakes"] = str(self.mistakes)
        game.headers["blunders"] = str(self.blunders)
        game.headers["forkes_and_pins"] = str(self.forks_and_pins)
        game.headers["optimal_moves"] = str(self.optimal_moves)
        game.headers["birilliants"]=str(self.brilliants)
        game.headers["miss"]=str(self.miss)
        game.headers["Result"] = self.result or "*"


        # Add moves to PGN
        node = game
        for move in self.moves:
            node = node.add_variation(move)

        # Save to file
        with open(filename,"a") as pgn_file:
            print(game, file=pgn_file)
            print("", file=pgn_file)
  
        

# Example usage
def play(name,id,level):
    game = ChessGame(level,id,name)
    fen_before=game.board.fen()
    
    fig, ax = setup_realtime_board_view()
    player_moves=[r"images\2-c5.JPG",r"images\4-d6.JPG",r"images\6-c4d5.JPG",r"images\8-kg6.JPG"]
    for i in range(9):
        if game.active_color == game.student_color:
            #image process---------------------------------------------------------------------
            if i%2==1:
                image=player_moves[i//2] 
            image = cv2.imread(image) #image=get_pict()

            squares=get_squares(image)

            fen_after=get_fen_from_board(squares,model_for_piece,model_for_color)

            move=get_move_from_fen(fen_before,fen_after)
            #--------------------------------------------------------------------------------


            game.make_move(move)

            rate=[0 for i in range(6)]
            comment=explain_move(fen_before,game.board.fen(), move,rate)
            game.update(rate)        

            fen_before=game.board.fen()
            
            show_board(ax, fen_before,comment)
            game.active_color = "White"
        else:
            ai_move = get_move(fen_before)  
            #to keep the same order of the images and not effect player's moves
            if i==0:
                ai_move=chess.Move.from_uci("e2e4")
            elif i==2:
                ai_move=chess.Move.from_uci("g1f3")    
            elif i==4:
                  ai_move=chess.Move.from_uci("d2d4") 
            elif i==6:
                ai_move=chess.Move.from_uci("f3d4")
            elif i==8:
                ai_move=chess.Move.from_uci("b1c3")    
            
            #________________________________________________________________
            board_before = chess.Board(fen_before)
            piece =  board_before.piece_at(ai_move.from_square)

            comment=f"i played {piece} to {ai_move.uci()[2:]}"
            game.make_move(ai_move)
            board_before = chess.Board(fen_before)
            piece =  board_before.piece_at(ai_move.from_square)
            fen_before = game.board.fen()

            show_board(ax, fen_before,comment)
            game.active_color = "Black"


    plt.ioff()
    plt.show()
    if game.board.is_checkmate():
        result="Checkmate! Game over.\n"
        if game.board.turn == chess.WHITE:
            result+="Black wins!"
            game.result="0-1"
        else:
            result+="White wins!"
            game.result="1-0"
    elif game.board.is_stalemate():
        result="Stalemate! Game over."   
        game.result="1/2-1/2"
    ax.set_title(result, fontsize=12, pad=10)
    plt.ioff()
    plt.show()#to keep the screen  
        

    return game


id_value, name, level,cluster,close= show_id_manager_interface()
if not close:
    while id_value==None:
        id_value, name, level,close= show_id_manager_interface()
        if close==True:
            break
    
print(f"ID: {id_value}, Name: {name}, Level: {level}")
game=play(name,id_value,level)
os._exit(0)
from time import sleep
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
import tkinter as tk
from tkinter import messagebox
import os


#-------------------------------------------------------------------------------------------------------------------------------------------   


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COUNTER_FILE = "game_counter.txt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

engine_path=r"stockfish\stockfish-windows-x86-64-avx2.exe"
engine = chess.engine.SimpleEngine.popen_uci(engine_path)


#-------------------------------------------------------------------------------------------------------------------------------------------    

def get_move(fen, level="pro"):
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
    while not game.board.is_game_over():
        if game.active_color == game.student_color:
            move=get_move(fen_before)#enter_move()
            #move=chess.Move.from_uci(move)
            
            game.make_move(move)
            #evaluate=[game.mistakes,game.miss,game.brilliants,game.blunders,game.forks_and_pins,game.optimal_moves]
            rate=[0 for i in range(6)]
            comment=explain_move(fen_before,game.board.fen(), move,rate)
            game.update(rate)        

            fen_before=game.board.fen()
            
            show_board(ax, fen_before,comment)
            game.active_color = game.arm_color
            
        else:
        
            ai_move = get_move(fen_before)  
            board_before = chess.Board(fen_before)
            piece =  board_before.piece_at(ai_move.from_square)

            comment=f"i played {piece} to {ai_move.uci()[2:]}"
            game.make_move(ai_move)
            board_before = chess.Board(fen_before)
            piece =  board_before.piece_at(ai_move.from_square)
            fen_before = game.board.fen()

            show_board(ax, fen_before,comment)
            game.active_color = game.student_color

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

    replay=True
    while replay:
        game=play(name,id_value,level)
        game.save_to_pgn("game.pgn")
        # replay=True
        replay=ask_to_play_another()  
os._exit(0)
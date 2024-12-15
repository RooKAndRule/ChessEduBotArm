from GUIPlayer import show_board,choose_color,show_id_manager_interface,ask_to_play_another,setup_realtime_board_view
import matplotlib.pyplot as plt
from comments import explain_move
# from code_Arm import move_piece,cleanup,get_pict

import torch
import matplotlib.pyplot as plt
import pickle
import torch
from torchvision import transforms
from PIL import Image
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


engine_path=r"stockfish\stockfish-windows-x86-64-avx2.exe"
engine = chess.engine.SimpleEngine.popen_uci(engine_path)

# def enter_move():
#     result = ""  

#     def get_value():
#         nonlocal result  
#         result = entry.get()  

#         root.destroy()  
#     root = tk.Tk()
#     root.title("Input Value")  
#     root.geometry("300x150")  

#     label = tk.Label(root, text="Enter a string:")
#     label.pack(pady=5)

#     entry = tk.Entry(root, width=30)
#     entry.pack(pady=5)

#     submit_button = tk.Button(root, text="Submit", command=get_value)
#     submit_button.pack(pady=10)

#     root.mainloop()

#     return result  

def evaluate_move( board, move_uci, time=1.0):

    move = chess.Move.from_uci(move_uci)

    if move in board.legal_moves:

        result = engine.analyse(board, chess.engine.Limit(time=time), moves=[move])
        evaluation_score = result["score"].relative.score  # Get evaluation score
        return evaluation_score
    else:
        return None


def load_model_and_tokenizer(id):
    model=None
    tokenizer=None
    if os.path.exists(f"models based on previous games\\players_models\\playerLSTMs{id}.h5"):
        model = load_model(f"models based on previous games\\players_models\\playerLSTMs{id}.h5")


        # Load the tokenizer
        with open(f"models based on previous games\\players_models\\playertokenizer{id}.pkl", "rb") as file:
            tokenizer = pickle.load(file)

    return model,tokenizer 

def lstm_predict(model, tokenizer, sequence):

    sequence_tokenized = tokenizer.texts_to_sequences([sequence])  
    sequence_padded = np.array(sequence_tokenized).reshape(1, -1) 
    

    predictions = model.predict(sequence_padded, verbose=0) 
    predicted_index = np.argmax(predictions)  
    return predicted_index, predictions

def compute_rl_feedback(predicted_move, actual_move,board,c):
    f={0:0.8,1:0.3,2:0.5}
    w1=f[c] if c !=-1 else 0.5
    w2=1-w1 if c !=-1 else 0.5,
    E1=evaluate_move(move_uci=predicted_move,board=board)
    E2=evaluate_move(move_uci=actual_move,board=board)

    S=1 if actual_move==predicted_move else 0

    R=w1*(E2-E1)+w2*S

    improvementFB="Your learning journey is progressing. Even though you're exploring new strategies, keep focusing on the lessons from past games."
    mistakeFB="We can adjust the lessons to match your learning style better. Stay focused and take your time."

    comment= improvementFB if E2>E1 else mistakeFB

    return R,comment


def train_lstm_with_reward(model, tokenizer, sequence, actual_move, reward):
    sequence_padded = np.array(sequence).reshape(1, -1)
    vocab_size = len(tokenizer.word_index) + 1
    y_true = np.zeros((1, vocab_size))

    if actual_move not in tokenizer.word_index:
        tokenizer.word_index[actual_move] = vocab_size
        vocab_size += 1
        y_true = np.zeros((1, vocab_size))
    
    target_index = tokenizer.word_index[actual_move]
    y_true[0, target_index] = reward

    model.fit(sequence_padded, y_true, epochs=1, verbose=0)


#-------------------------------------------------------------------------------------------------------------------------------------------    

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


    for option, value in options.items():
        engine.configure({option: value})

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


    def save_to_pgn(self, filename="games.pgn"):
        
        game = chess.pgn.Game()
        game.headers["Event"] = "TSPY12"
        game.headers["Site"] = "local"
        game.headers["Date"] = datetime.date.today().strftime("%Y-%m-%d %H:%M:%S")
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

        
      

# Example usage
def play(name,id,level,cluster):
    model,tokenizer=load_model_and_tokenizer(id)
    print("Tokenizer Vocabulary:", tokenizer.word_index)

    game = ChessGame(level,id,name)
    fen_before=game.board.fen()

    game_sequence = []
    
    fig, ax = setup_realtime_board_view()
    while not game.board.is_game_over():
        if game.active_color == game.student_color:
            move=get_move(fen_before)#enter_move()
            #move=chess.Move.from_uci(move)

            if len(game_sequence)>0:
                predicted_move, prediction = lstm_predict(model,tokenizer,game_sequence[-30:])
                reward,comment = compute_rl_feedback(predicted_move, str(move),cluster)
                train_lstm_with_reward(model,tokenizer,game_sequence[-30:], str(move),reward)



            game.make_move(move)
            game_sequence.append(str(move))
            #evaluate=[game.mistakes,game.miss,game.brilliants,game.blunders,game.forks_and_pins,game.optimal_moves]
            rate=[0 for i in range(6)]
            comment=explain_move(fen_before,game.board.fen(), move,rate)
            game.update(rate)        

            fen_before=game.board.fen()
            
            show_board(ax, fen_before,comment)
            game.active_color = game.arm_color
            
        else:
        
            ai_move = get_move(fen_before)  

            game_sequence.append(str(ai_move))
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
    elif game.board.is_insufficient_material():
        result = "Draw due to insufficient material."
        game.result = "1/2-1/2"
    elif game.board.is_seventyfive_moves():
        result = "Draw due to the seventy-five-move rule."
        game.result = "1/2-1/2"
    elif game.board.is_fivefold_repetition():
        result = "Draw due to fivefold repetition."
        game.result = "1/2-1/2"    
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
        game=play(name,id_value,level,cluster)
        game.save_to_pgn("game.pgn")
        # replay=True
        replay=ask_to_play_another()  
os._exit(0)
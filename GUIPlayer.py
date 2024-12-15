from datetime import datetime
import tkinter as tk
import chess
import chess.svg
import matplotlib.pyplot as plt
import cairosvg
import io
from tkinter import messagebox


def center_window(window, width, height):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)
    window.geometry(f"{width}x{height}+{x}+{y}")

#---------color ?---------------------------------------------------------------------------------------------------------------
def choose_color():
    def on_button_click(color):
        nonlocal chosen_color
        chosen_color = color
        wind.destroy()  

    wind = tk.Tk()
    wind.title("Choose Your Color")
    center_window(wind, 300, 150)  

    chosen_color = None

    question_label = tk.Label(wind, text="What color will you play?", font=("Arial", 14))
    question_label.pack(pady=20)

    black_button = tk.Button(wind, text="Black", font=("Arial", 12), bg="black", fg="white", width=10, 
                             command=lambda: on_button_click("Black"))
    white_button = tk.Button(wind, text="White", font=("Arial", 12), bg="white", fg="black", width=10, 
                             command=lambda: on_button_click("White"))

    black_button.pack(side=tk.LEFT, padx=30)
    white_button.pack(side=tk.RIGHT, padx=30)

    wind.mainloop()

    return chosen_color

#---------board plot---------------------------------------------------------------------------------------------------------------

def setup_realtime_board_view():

    fig, ax = plt.subplots()
    ax.axis('off') 
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.ion() 
    return fig, ax


def show_board(ax, fen,comment=None):

    board = chess.Board(fen)
    svg = chess.svg.board(board)
    png = cairosvg.svg2png(bytestring=svg)

    image_stream = io.BytesIO(png)
    img = plt.imread(image_stream, format="png")

    ax.clear()
    ax.axis('off')
    ax.imshow(img)

    if comment:
        ax.set_title(comment, fontsize=12, pad=10)

    plt.draw()
    plt.pause(0.1)  #visualization




#---------Login student---------------------------------------------------------------------------------------------------------------
def verify_id(entry_id, result):
    id_value = entry_id.get()
    if not id_value:
        messagebox.showwarning("Warning", "Please enter an ID.")
        return

    try:
        with open("ids.txt", "r") as file:
            ids = file.read().splitlines()
            found = False
            for record in ids:
                if record.startswith(f"ID:{id_value}"):

                    _, name, age, level ,cluster,_= record.split(', ')
                    name = name.split(':')[1]
                    age = age.split(':')[1]
                    level = level.split(':')[1]
                    cluster = cluster.split(':')[1]

                    # Store results
                    result["id"] = id_value
                    result["name"] = name
                    result["level"] = level
                    result["cluster"]=cluster

                    messagebox.showinfo(
                        "Info", f"ID '{id_value}' exists.\nName: {name}\nAge: {age}\nLevel: {level}"
                    )
                    found = True
                    return
            if not found:
                messagebox.showinfo("Info", f"ID '{id_value}' does not exist in the file.")
    except FileNotFoundError:
        messagebox.showerror("Error", "The file 'ids.txt' does not exist.")


def save_player(entry_name, entry_age, level_var, cluster,last_clustering,result):
    name_value = entry_name.get()
    age_value = entry_age.get()
    level_value = level_var.get()

    if not name_value or not age_value or not level_value:
        messagebox.showwarning("Warning", "Please fill in all fields.")
        return

    try:
        with open("ids.txt", "r") as file:
            ids = file.read().splitlines()
            last_id = int(ids[-1].split(':')[1].split(',')[0]) if ids else 0
    except FileNotFoundError:
        last_id = 0

    player_id = last_id + 1
    with open("ids.txt", "a") as file:
        file.write(f"ID:{player_id}, Name:{name_value}, Age:{age_value}, Level:{level_value}, cluster:{cluster}, last_clustering:{last_clustering}\n")


    result["id"] = str(player_id)
    result["name"] = name_value
    result["level"] = level_value
    
    # ------------------------------------------------
    result["cluster"] = str(cluster)  
    result["last_clustering"] =str(last_clustering)
    # ------------------------------------------------

    messagebox.showinfo(
        "Success", f"Player '{name_value}' has been added with ID '{player_id}'."
    )


def open_new_player_window(result):
    new_player_window = tk.Toplevel()
    new_player_window.title("New Player")
    center_window(new_player_window, 400, 250) 

    tk.Label(new_player_window, text="Enter Name:").pack(pady=5)
    entry_name = tk.Entry(new_player_window, width=30)
    entry_name.pack(pady=5)

    tk.Label(new_player_window, text="Enter Age:").pack(pady=5)
    entry_age = tk.Entry(new_player_window, width=30)
    entry_age.pack(pady=5)

    tk.Label(new_player_window, text="Select Level:").pack(pady=5)
    level_var = tk.StringVar(value="")
    levels = ["Beginner", "Intermediate", "Expert"]
    tk.OptionMenu(new_player_window, level_var, *levels).pack(pady=5)

    tk.Button(
        new_player_window,
        text="Save Player",
        command=lambda: [save_player(entry_name, entry_age, level_var,-1,datetime.now().strftime("%Y-%m-%d %H:%M:%S"),result), new_player_window.destroy()]
    ).pack(pady=10)


def show_id_manager_interface():
    def on_button_click(closed):
        nonlocal close
        close = closed
        wind.destroy()

    close = False
    result = {"id": None, "name": None, "level": None,"cluster":None}

    # Main application window
    wind = tk.Tk()
    wind.title("ID Manager")
    center_window(wind, 400, 300)  # Center the window

    tk.Label(wind, text="Enter your ID:").pack(pady=5)
    entry_id = tk.Entry(wind, width=30)
    entry_id.pack(pady=5)

    # Buttons
    tk.Button(
        wind,
        text="play",
        command=lambda: [verify_id(entry_id, result), wind.destroy()]
    ).pack(pady=5)

    tk.Button(
        wind,
        text="New Player",
        command=lambda: open_new_player_window(result)
    ).pack(pady=5)


    tk.Button(
        wind,
        text="Disconnect",
        command=lambda: on_button_click(True)
    ).pack(pady=5)

    wind.mainloop()


    return result["id"], result["name"], result["level"],result["cluster"], close

#----------other game?------------------------------------------------------------------------------------------------------------------------------
def on_yes(popup, choice):
    choice[0] = True
    popup.destroy()

def on_no(popup, choice):
    choice[0] = False
    popup.destroy()

def ask_to_play_another():
    choice = [None]  

    popup = tk.Tk()
    popup.title("Play Again")
    center_window(popup, 300, 150) 

    tk.Label(popup, text="Do you want to play another game?", font=("Arial", 12)).pack(pady=10)

    btn_yes = tk.Button(popup, text="Yes", command=lambda: on_yes(popup, choice), width=10)
    btn_no = tk.Button(popup, text="No", command=lambda: on_no(popup, choice), width=10)

    btn_yes.pack(side=tk.LEFT, padx=20)
    btn_no.pack(side=tk.RIGHT, padx=20)

    popup.wait_window()  
    return choice[0]

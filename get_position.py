import torch
from torchvision import transforms
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

# Set parameters
IMG_HEIGHT = 181
IMG_WIDTH = 181
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_transforms_for_piece = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert to grayscale
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)), 
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize the tensor
])

def predict_image(model, square, transform, class_names):
    model.eval()  
    img = Image.fromarray(square)  
    img = img.convert("RGB")  
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)  # Apply transformations and move to device

    with torch.no_grad():  # Disable gradient calculation
        output = model(img_tensor) 
        _, pred = torch.max(output, 1)  # Get the predicted class
    
    return class_names[pred.item()]  

# Example usage
#piece_list = ['p', 'b', 'b', ' ', 'k', 'p', ' ', 'p', 'p', 'p', ' ', 'p', ' ', 'p', 'p', 'p', ' ', ' ', 'n', ' ', 'p', 'n', ' ', ' ', ' ', 'N', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'P', 'P', ' ', ' ', ' ', ' ', 'N', ' ', ' ', ' ', 'P', ' ', 'P', 'P', 'P', ' ', ' ', 'P', ' ', 'P', 'R', ' ', ' ', 'Q', 'K', 'B', ' ', 'R']

def list_to_fen(piece_list):
    fen = ""
    empty_count = 0

    for i, piece in enumerate(piece_list):
        if piece == ' ':
            empty_count += 1
        else:
            if empty_count > 0:
                fen += str(empty_count)
                empty_count = 0
            fen += piece
        
        # Add a slash every 8 elements (for 8 squares in a row)
        if (i + 1) % 8 == 0:
            if empty_count > 0:
                fen += str(empty_count)
                empty_count = 0
            if i != len(piece_list) - 1:  # Avoid adding a slash at the end
                fen += '/'
    
    return fen




def get_fen_from_board(sqaures,model_for_piece,model_for_color):
    pieces_classes = ['Bishop',
                      'King',
                      'Knight',
                      'Pawn',
                      'Queen',
                      'Rook',
                      'empty'] 
    predictions_pieces = [] 

    for i in range(64):
        img = sqaures[i]  
        predicted_class = predict_image(model_for_piece,
                                        img, val_transforms_for_piece, 
                                        pieces_classes)  # Predict the piece class
        predictions_pieces.append(predicted_class)  
        

    Color=["black","white"]
    predictions_colors=[]

    for i in range(64):
            img = sqaures[i]
            if predictions_pieces[i]!="empty":
                    pred=predict_image(model_for_color, img, val_transforms, Color)
                    predictions_colors.append(pred)
            else :
                    predictions_colors.append("empty")    

    pieces=[]
    for i in range(64):
        if predictions_pieces[i]=="Knight":
            predictions_pieces[i]="Night"
        if predictions_colors[i]=="black":
            pieces.append(predictions_pieces[i][0].lower())
        elif predictions_colors[i]=="white":   
            pieces.append(predictions_pieces[i][0].upper())
        else :
            pieces.append(" ")          
    fen=list_to_fen(pieces)                      

    return fen
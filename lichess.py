import requests

import json

import pandas as pd

import statistics as sta

def fetch_games(username, max_games=100):
    # URL de l'API Lichess pour récupérer les parties
    url = f"https://lichess.org/api/games/user/{username}"
    headers = {
        'Accept': 'application/x-ndjson',
    }

    params = {
        'max': max_games,
        'pgnInJson': True,  # Pour recevoir les PGN annotés
        'withTimestamps': True,  # Pour obtenir les temps par coup
        'clocks': 1,
        'evals': 1,  # Inclure les évaluations après chaque coup
        'opening': 1,
    }

    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()

    return response.text

def save_games_to_pgn(games_data, filename):
    with open(filename, 'w') as f:
        f.write(games_data)
'''
def main():
    username = input("Entrez le nom d'utilisateur de Lichess: ")  #EricRosen
    max_games = int(input("Entrez le nombre maximum de parties à récupérer: "))

    print(f"Récupération des parties pour l'utilisateur {username}...")
    games_data = fetch_games(username, max_games)

    # Enregistrement des parties dans un fichier PGN
    filename = f"{username}_games.pgn"
    save_games_to_pgn(games_data, filename)

    print(f"Les parties ont été sauvegardées dans le fichier {filename}")

if __name__ == "__main__":
    main()
'''


def tab_info_games(username, max_games):          #renvoi une tableau tab tel a que tab[i] est un tableau contenant toutesles infos intéressantes de la partie numéro i (dans l'ordre): le type de la partie (souvent blitz ou bullet), le nom d'utilisateur jouant les blancs, le nom d'utilisateur jouant les noirs, le vainqueur, la manière de gagner, elo blanc, elo noir, ouverture, les coups, les evaluations avant chaque coups, les temps  
  games_data=fetch_games(username, max_games)
  #print(games_data)
  tab = []

  for line in games_data.splitlines():
        game = json.loads(line)  # Convertir chaque ligne JSON en dictionnaire
        game_info = {
            'type': game.get('speed', 'Unknown'),
            'white_player': game['players']['white']['user']['name'] if 'user' in game['players']['white'] else 'Anonymous',
            'black_player': game['players']['black']['user']['name'] if 'user' in game['players']['black'] else 'Anonymous',
            'winner': game.get('winner', 'draw'),
            'termination': game.get('status', 'Unknown'),
            'white_elo': game['players']['white'].get('rating', 'Unknown'),
            'black_elo': game['players']['black'].get('rating', 'Unknown'),
            'opening': game['opening'].get('name', 'Unknown') if 'opening' in game else 'Unknown',
            'moves': game.get('moves', '').split(),
            'evaluations': [analysis.get('eval', 'N/A') for analysis in game.get('analysis', [])],
            'clocks': game.get('clocks', [])
        }
        tab.append(game_info)
    
  return tab

#print(tab_info_games("EricRosen",4))

def info_player (username,max_games): #A very ugly function that gives back a panda data frame whith computable data for each game (agregated data from clock and evaluations rather than lists)
    info=tab_info_games(username, max_games)
    table=[]
    for i in range(max_games):
        table.append(list(pd.DataFrame(info[i].items())[1])) #get a table with the data from the dictionnary
    #initiate variables
    m_o_r=[]
    m_m_1_r=[]
    m_m_2_r=[]
    m_e_1_r=[]
    m_e_2_r=[]
    m_o_t=[]
    m_m_1_t=[]
    m_m_2_t=[]
    m_e_1_t=[]
    m_e_2_t=[]
    Len_game=[] 
    won=[]
    elo_diff=[]
    white=[]

    for i in range(max_games): #Compute the variable won that is 1 if the player won or 0 otherwise,  elo diff, the difference of elo, and white, that is 1 if the player plays first
        if table[i][1]==username :
            white.append(1)
            elo_diff.append(table[i][5]-table[i][6])
            if table[i][3]=='white' :
                won.append(1)
            else : won.append(0)
        else :
            white.append(0)
            elo_diff.append(table[i][6]-table[i][5])
            if table[i][3]=='black' :
                won.append(1)
            else : won.append(0)

    for i in range(max_games): #Get two lists the times and evaluations specific to each move that the player made
        evt2=table[i][-2]
        cl2=table[i][-1]
        n=len(evt2)
        if white[i]==1 : #if player plays first get the even moves
            evt=evt2[::2]
            cl=cl2[::2]
        else: #otherwise get the odd moves
            evt=evt2[1::2]
            cl=cl2[1::2]
        ev= []
        for j in range(1,len(evt)): #get the individual rating of each move
            if not isinstance(evt[j],int) or not isinstance(evt[j-1],int) :
                ev.append(0)
            else :
                ev.append(evt[j] - evt[j-1])

        if len(ev)>41 : 
            indices=[(0,10), (11,20), (21,30),(31,40),(41,n)] #split the moves in five categories : opening, middle game1, middle game 2, endgame1, endgame2 with ten move increments. Endgame2 has the remaining moves
            means_ev=[sta.mean(ev[a:b+1]) for a,b in indices] #compute for each slice the mean rating of the move
            means_time=[sta.mean(cl[a:b+1]) for a,b in indices] #compute for each slice the mean time taken for the move
            Len_game.append("Long") #Get the length of the game based on the number of moves
        elif len(ev)>31 :
            indices=[(0,10), (11,20),(21,30),(31,n)]
            means_ev=[sta.mean(ev[a:b+1]) for a,b in indices]
            means_time=[sta.mean(cl[a:b+1]) for a,b in indices]
            means_time.append('NaN') #NaN is added if there wasn't enough moves in the game to complete the last slices
            means_ev.append('NaN')
            Len_game.append("Medium")
        elif len(ev)>21 :
            indices=[(0,10), (11,20),(21,n)]
            means_ev=[sta.mean(ev[a:b+1]) for a,b in indices]
            means_time=[sta.mean(cl[a:b+1]) for a,b in indices]
            means_time+=['NaN','NaN']
            means_ev+=['NaN','NaN']
            Len_game.append("Short")
        else : #If the game had less than 21 moves the nalysis in terms of opening, middle game, is not useful : nothing is computed
            means_ev=['Nan' for k in range(5)]
            means_time=['Nan' for k in range(5)]
            Len_game.append("Too Short")
        m_o_r.append(means_ev[0]) #Then the computed measures are added
        m_m_1_r.append(means_ev[1])
        m_m_2_r.append(means_ev[2])
        m_e_1_r.append(means_ev[3])
        m_e_2_r.append(means_ev[4])
        m_o_t.append(means_time[0])
        m_m_1_t.append(means_time[1])
        m_m_2_t.append(means_time[2])
        m_e_1_t.append(means_time[3])
        m_e_2_t.append(means_time[4])
    Data = pd.DataFrame(table,columns=['type','white_player','black_player','winner','termination','white_elo','black_elo','opening','moves','evaluations','clocks'])
    Data["white"]=white
    Data["won"]=won
    Data["length_game"]=Len_game
    Data["elo_diff"]=elo_diff
    Data["mean_opening_rating"]=m_o_r #the new colums are added to the data set
    Data["mean_middle1_rating"]=m_m_1_r
    Data["mean_middle2_rating"]=m_m_2_r
    Data["mean_end1_rating"]=m_e_1_r
    Data["mean_end2_rating"]=m_e_2_r
    Data["mean_opening_time"]=m_o_t
    Data["mean_middle1_time"]=m_m_1_t
    Data["mean_middle2_time"]=m_m_2_t
    Data["mean_end1_time"]=m_e_1_t
    Data["mean_end2_time"]=m_e_2_t
    Data=Data.drop(['white_player','black_player','winner','white_elo','black_elo','moves','evaluations','clocks'],axis=1) #The non computable variables are dropped from the data set
    return Data

print(info_player("EricRosen",20))

username = "EricRosen"

print(tab_info_games(username,1))

Board=[(a,b) for a in [11,12,13,14,15,16,17,18] for b in [1,2,3,4,5,6,7,8]]

Pieces=[('WRa',5),('WPa',1),0,0,0,0,('BPa',1),('BRa',5),
('WNb',3),('WPb',1),0,0,0,0,('BPb',1),('BNb',3), 
('WBc',3),('WPc',1),0,0,0,0,('BPc',1),('BBc',3),
('WQ',9),('WPd',1),0,0,0,0,('BPd',1),('BQ',9),
('WK',0),('WPe',1),0,0,0,0,('BPe',1),('BK',0),
('WBf',3),('WPf',1),0,0,0,0,('BPf',1),('BBf',3),
('WNg',3),('WPg',1),0,0,0,0,('BPg',1),('BNg',3),
('WRh',5),('WPh',1),0,0,0,0,('BPh',1),('BRh',5)]

Initial_board={}

k=0
for j in Board :
    Initial_board[j]=Pieces[k]
    k+=1

print(Initial_board)

Taken=[]

def convert(a,b) :
    if a == 'a' : return (11,int(b))
    elif a == 'b' : return (12,int(b))
    elif a == 'c' : return (13,int(b))
    elif a == 'd' : return (14,int(b))
    elif a == 'e' : return (15,int(b))
    elif a == 'f' : return (16,int(b))
    elif a == 'g' : return (17,int(b))
    elif a == 'h' : return (18,int(b))

def pieces_taken (moves) :
    board=copy.deepcopy(Initial_board)
    turn=1
    for i in moves :
        if i.find('+')!=-1:
            mo=[*i].pop()
        else :
            mo=[*i]
        new_p=convert(mo[-2],mo[-1])
        if turn%2==1 :
            if len(mo)==2 :
                if i.find('x')==-1 :
                    board[new_p]= ('WP'+mo[-2],1)
                    if board[(new_p[0],new_p[1]-1)]==0 :
                        board[(new_p[0],new_p[1]-2)]=0
                    else : board[(new_p[0],new_p[1]-1)]=0
                else :
                    Taken.append([board[new_p],turn])
                    board[new_p]= ('WP'+mo[-2],1)
                    board[convert([*i][0],new_p[1]-1)]=0
        else :
            if len(mo)==2 :
                if i.find('x')==-1 :
                    board[new_p]= ('BP'+mo[-2],1)
                    if board[(new_p[0],new_p[1]+1)]==0 :
                        board[(new_p[0],new_p[1]+2)]=0
                    else : board[(new_p[0],new_p[1]+1)]=0
                else :
                    Taken.append([board[new_p],turn])
                    board[new_p]= ('BP'+mo[-2],1)
                    board[convert([*i][0],new_p[1]+1)]=0
        turn+=1
    return board

print(pieces_taken(['d4', 'd5', 'c4', 'e6']))
        



#def user_evaluations_list(n): # n-th game
    #if tab_info_games(username,5)[n]['white_player'] == username:
        # even indices
        #evaluations_list = tab_info_games(username,5)[n]['evaluations'][0::2]
    #else:
        # odd indices
        #evaluations_list = tab_info_games(username,5)[n]['evaluations'][1::2]
    #return evaluations_list

#def move_evaluations_list(n):
    #u_list = user_evaluations_list(n)
    #m_list = [u_list[0]]
    #for i in range(1, len(u_list) - 1):
        #m_list.append(u_list[i] - u_list[i-1])
    #return m_list

#move_evaluations_list(0)


#def user_clocks_list(n): # n-th game
    #clocks_time=[(tab_info_games("EricRosen",5)[n]['clocks'][i]-tab_info_games("EricRosen",5)[n]['clocks'][i+1]) for i in (len(tab_info_games("EricRosen",5)[n]['clocks'])-1)]   #marche pas
    #if tab_info_games(username,5)[n]['white_player'] == username:
        # even indices
        #clocks_list = clocks_time[0::2]
    #else:
        # odd indices
        #clocks_list = clocks_time[1::2]
    #return clocks_list

#user_clocks_list(0)

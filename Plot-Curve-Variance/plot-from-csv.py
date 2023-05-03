import csv
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(__file__) + os.sep + './')


def read_csv_2_dict(csv_str, step=1):
    csv_reader = csv.reader(open(csv_str))
    i = 0
    list_all = []
    for row in csv_reader:
        if i > 0 and i % step == 0:
            list_all.append([int(row[1]), float(row[2])])
        i += 1
    return list_all


def smoothen(data, w):
    res = np.zeros_like(data)
    for i in range(len(data)):
        if i > w:
            res[i] = np.mean(data[i-w:i])
        elif i > 0:
            res[i] = np.mean(data[:i])
        else:  # i == 0
            res[i] = data[i]
    return res


def draw(data_dict, i, w):
    color = ['orange', 'hotpink', 'dodgerblue', 'mediumpurple', 'c', 'cadetblue', 'steelblue', 'mediumslateblue',
             'hotpink', 'mediumturquoise']

    """
    color = [
        'aqua',
        'aquamarine',
        'bisque',
        'black',
        'blue',
        'blueviolet',
        'brown',
        'burlywood',
        'cadetblue',
        'chartreuse',
        'chocolate',
        'coral',
        'cornflowerblue',
        'crimson',
        'cyan',
        'darkblue',
        'darkcyan',
        'darkgoldenrod',
        'darkgray',
        'darkgreen',
        'darkkhaki',
        'darkmagenta',
        'darkolivegreen',
        'darkorange',
        'darkorchid',
        'darkred',
        'darksalmon',
        'darkseagreen',
        'darkslateblue',
        'darkslategray',
        'darkturquoise',
        'darkviolet',
        'deeppink',
        'deepskyblue',
        'dimgray',
        'dodgerblue',
        'firebrick',
        'floralwhite',
        'forestgreen',
        'fuchsia',
        'gainsboro',
        'ghostwhite',
        'gold',
        'goldenrod',
        'gray',
        'green',
        'greenyellow',
        'honeydew',
        'hotpink',
        'indianred',
        'indigo',
        'ivory',
        'khaki',
        'lavender',
        'lavenderblush',
        'lawngreen',
        'lemonchiffon',
        'lightblue',
        'lightcoral',
        'lightcyan',
        'lightgoldenrodyellow',
        'lightgreen',
        'lightgray',
        'lightpink',
        'lightsalmon',
        'lightseagreen',
        'lightskyblue',
        'lightslategray',
        'lightsteelblue',
        'lightyellow',
        'lime',
        'limegreen',
        'linen',
        'magenta',
        'maroon',
        'mediumaquamarine',
        'mediumblue',
        'mediumorchid',
        'mediumpurple',
        'mediumseagreen',
        'mediumslateblue',
        'mediumspringgreen',
        'mediumturquoise',
        'mediumvioletred',
        'midnightblue',
        'mintcream',
        'mistyrose',
        'moccasin',
        'navajowhite',
        'navy',
        'oldlace',
        'olive',
        'olivedrab',
        'orange',
        'orangered',
        'orchid',
        'palegoldenrod',
        'palegreen',
        'paleturquoise',
        'palevioletred',
        'papayawhip',
        'peachpuff',
        'peru',
        'pink',
        'plum',
        'powderblue',
        'purple',
        'red',
        'rosybrown',
        'royalblue',
        'saddlebrown',
        'salmon',
        'sandybrown',
        'seagreen',
        'seashell',
        'sienna',
        'silver',
        'skyblue',
        'slateblue',
        'slategray',
        'snow',
        'springgreen',
        'steelblue',
        'tan',
        'teal',
        'thistle',
        'tomato',
        'turquoise',
        'violet',
        'wheat',
        'white',
        'whitesmoke',
        'yellow',
        'yellowgreen']
        """
    plt.xlabel("Environment steps", fontsize=18)
    plt.ylabel("Average Episode Cost", fontsize=18) # Reward Cost

    for k, episode_rewards in data_dict.items():
        timestep = np.array(episode_rewards)[:, :, 0][0]
        if scenario == "HumanoidStandup":
            reward = np.array(episode_rewards)[:, :, 1] / 1000
        else:
            reward = np.array(episode_rewards)[:, :, 1]
        r_mean, r_std = np.mean(reward, axis=0), np.std(reward, axis=0, ddof=1)

        r_mean = smoothen(r_mean, w)
        r_std = smoothen(r_std, w)
        print("r_mean - r_std", r_mean - r_std)

        plt.plot(timestep, r_mean, color=color[i], label=k, linewidth=1.5)
        plt.fill_between(timestep, r_mean - r_std, r_mean + r_std, alpha=0.2, color=color[i])
        i += 1


if __name__ == "__main__":
    # scenario = "Ant"
    # config = "2x4"
    scenario = "ManyAgent Ant"
    config = "6x1"
    performance_type = "costs" #"costs" # "rewards"
    smoothen_w = 5
    need_legend = True  # False True
    # need_legend = True
    if scenario == "Reacher":
        plt.ylim(ymin=-200, ymax=0)
    if scenario == "ManyAgentSwimmer":
        plt.ylim(ymin=-150, ymax=250)
    if scenario == "HumanoidStandup":
        plt.text(-1e6, 145, r'1e3', fontsize=10)

    data_dict = {}

    episode_rewards = []
    for i in range(3):
        index = i + 1
        csv_path = "./sadppo/manyagent/manyagent" + config + "_lr9e5_3seeds/" + performance_type + "_seed" + str(index) + ".csv"
        list_ = read_csv_2_dict(csv_path)
        episode_rewards.append(list_)
    data_dict['HAPPO'] = episode_rewards

    episode_rewards = []
    for i in range(3):
        index = i + 1
        csv_path = "./wu_mappo/manyagent_ant/manyagent"+ config +"_lr9e5_3seeds/" + performance_type + "_seed" + str(index) + ".csv"
        list_ = read_csv_2_dict(csv_path)
        episode_rewards.append(list_)
    data_dict['MAPPO'] = episode_rewards

    # episode_rewards = []
    # for i in range(3):
    #     index = i + 1 #matrpo/manyagent_ant/ruqing_kl0.01_manyant_2x3
    #     csv_path = "./matrpo/manyagent_ant/ruqing_kl0.01_manyant_" + config + "/" + performance_type + "_seed" + str(
    #         index) + ".csv"
    #     list_ = read_csv_2_dict(csv_path)
    #     episode_rewards.append(list_)
    # data_dict['HATRPO'] = episode_rewards
    #
    episode_rewards = []
    for i in range(3):
        index = i + 1 #~/plot_macppo_results/mactrpo/ManyAgent_Ant_new/manyagent2x3_lr9e5_3seeds
        csv_path = "./mactrpo/ManyAgent_Ant_new/manyagent" + config + "_lr9e5_3seeds/" + performance_type + "_seed" + str(
            index) + ".csv"
        list_ = read_csv_2_dict(csv_path)
        episode_rewards.append(list_)
    data_dict['MACPO (ours)'] = episode_rewards

    episode_rewards = []
    for i in range(3):
        index = i + 1
        csv_path = "./mappo_lagr/manyagent_ant/manyant"+ config +"_lr9e5_exp1_3seeds/" + performance_type + "_seed" + str(index) + ".csv"
        list_ = read_csv_2_dict(csv_path)
        episode_rewards.append(list_)
    data_dict['MAPPO-L (ours)'] = episode_rewards

    csv_path = "./maddpg/Half_Cheetah/Half_Cheetah" + config + "/" + performance_type + "_seed" + str(index) + ".csv"

    episode_rewards = []
    for i in range(3):
        index = i + 1
        csv_path = "./ippo/manyagent/manyagent" + config + "_3seeds/" + performance_type + "_seed" + str(
            index) + ".csv"
        list_ = read_csv_2_dict(csv_path)
        episode_rewards.append(list_)
    data_dict['IPPO'] = episode_rewards


    """
    episode_rewards = []
    for i in range(1):
        index = i + 1
        csv_path = "./mappo_lagr_ablation/manyant_" + config + "/mappo_lagrdot05_lagrCoef1e3/" + performance_type + "_seed" + str(
            index) + ".csv"
        list_ = read_csv_2_dict(csv_path)
        episode_rewards.append(list_)
    data_dict['Safe HAPPO--lagr-coef=0.05'] = episode_rewards

    episode_rewards = []
    for i in range(1):
        index = i + 1
        csv_path = "./mappo_lagr_ablation/manyant_" + config + "/mappo_lagrdot1_lagrCoef1e3/" + performance_type + "_seed" + str(
            index) + ".csv"
        list_ = read_csv_2_dict(csv_path)
        episode_rewards.append(list_)
    data_dict['Safe HAPPO--lagr-coef=0.1'] = episode_rewards

    episode_rewards = []
    for i in range(1):
        index = i + 1
        csv_path = "./mappo_lagr_ablation/manyant_" + config + "/mappo_lagrdot15_lagrCoef1e3/" + performance_type + "_seed" + str(
            index) + ".csv"
        list_ = read_csv_2_dict(csv_path)
        episode_rewards.append(list_)
    data_dict['Safe HAPPO--lagr-coef=0.15'] = episode_rewards

    episode_rewards = []
    for i in range(1):
        index = i + 1
        csv_path = "./mappo_lagr_ablation/manyant_"+ config +"/mappo_lagrdot2_lagrCoef1e3/"  + performance_type + "_seed" + str(
            index) + ".csv"
        list_ = read_csv_2_dict(csv_path)
        episode_rewards.append(list_)
    data_dict['Safe HAPPO--lagr-coef=0.2'] = episode_rewards

    episode_rewards = []
    for i in range(1):
        index = i + 1
        csv_path = "./mappo_lagr_ablation/manyant_" + config + "/mappo_lagrdot25_lagrCoef1e3/" + performance_type + "_seed" + str(
            index) + ".csv"
        list_ = read_csv_2_dict(csv_path)
        episode_rewards.append(list_)
    data_dict['Safe HAPPO--lagr-coef=0.25'] = episode_rewards

    episode_rewards = []
    for i in range(1):
        index = i + 1
        csv_path = "./mappo_lagr_ablation/manyant_" + config + "/mappo_lagrdot3_lagrCoef1e3/" + performance_type + "_seed" + str(
            index) + ".csv"
        list_ = read_csv_2_dict(csv_path)
        episode_rewards.append(list_)
    data_dict['Safe HAPPO--lagr-coef=0.3'] = episode_rewards

    episode_rewards = []
    for i in range(1):
        index = i + 1
        csv_path = "./mappo_lagr_ablation/manyant_" + config + "/mappo_lagrdot35_lagrCoef1e3/" + performance_type + "_seed" + str(
            index) + ".csv"
        list_ = read_csv_2_dict(csv_path)
        episode_rewards.append(list_)
    data_dict['Safe HAPPO--lagr-coef=0.35'] = episode_rewards

    episode_rewards = []
    for i in range(1):
        index = i + 1
        csv_path = "./mappo_lagr_ablation/manyant_" + config + "/mappo_lagrdot4_lagrCoef1e3/" + performance_type + "_seed" + str(
            index) + ".csv"
        list_ = read_csv_2_dict(csv_path)
        episode_rewards.append(list_)
    data_dict['Safe HAPPO--lagr-coef=0.4'] = episode_rewards

    episode_rewards = []
    for i in range(1):
        index = i + 1
        csv_path = "./mappo_lagr_ablation/manyant_" + config + "/mappo_lagrdot45_lagrCoef1e3/" + performance_type + "_seed" + str(
            index) + ".csv"
        list_ = read_csv_2_dict(csv_path)
        episode_rewards.append(list_)
    data_dict['Safe HAPPO--lagr-coef=0.45'] = episode_rewards

    episode_rewards = []
    for i in range(1):
        index = i + 1
        csv_path = "./mappo_lagr_ablation/manyant_" + config + "/mappo_lagrdot5_lagrCoef1e3/" + performance_type + "_seed" + str(
            index) + ".csv"
        list_ = read_csv_2_dict(csv_path)
        episode_rewards.append(list_)
    data_dict['Safe HAPPO--lagr-coef=0.5'] = episode_rewards

    episode_rewards = []
    for i in range(1):
        index = i + 1
        csv_path = "./mappo_lagr_ablation/manyant_" + config + "/mappo_lagrdot55_lagrCoef1e3/" + performance_type + "_seed" + str(
            index) + ".csv"
        list_ = read_csv_2_dict(csv_path)
        episode_rewards.append(list_)
    data_dict['Safe HAPPO--lagr-coef=0.55'] = episode_rewards

    episode_rewards = []
    for i in range(1):
        index = i + 1
        csv_path = "./mappo_lagr_ablation/manyant_" + config + "/mappo_lagrdot6_lagrCoef1e3/" + performance_type + "_seed" + str(
            index) + ".csv"
        list_ = read_csv_2_dict(csv_path)
        episode_rewards.append(list_)
    data_dict['Safe HAPPO--lagr-coef=0.6'] = episode_rewards

    episode_rewards = []
    for i in range(1):
        index = i + 1
        csv_path = "./mappo_lagr_ablation/manyant_" + config + "/mappo_lagrdot65_lagrCoef1e3/" + performance_type + "_seed" + str(
            index) + ".csv"
        list_ = read_csv_2_dict(csv_path)
        episode_rewards.append(list_)
    data_dict['Safe HAPPO--lagr-coef=0.65'] = episode_rewards

    episode_rewards = []
    for i in range(1):
        index = i + 1
        csv_path = "./mappo_lagr_ablation/manyant_" + config + "/mappo_lagrdot7_lagrCoef1e3/" + performance_type + "_seed" + str(
            index) + ".csv"
        list_ = read_csv_2_dict(csv_path)
        episode_rewards.append(list_)
    data_dict['Safe HAPPO--lagr-coef=0.7'] = episode_rewards

    episode_rewards = []
    for i in range(1):
        index = i + 1
        csv_path = "./mappo_lagr_ablation/manyant_" + config + "/mappo_lagrdot75_lagrCoef1e3/" + performance_type + "_seed" + str(
            index) + ".csv"
        list_ = read_csv_2_dict(csv_path)
        episode_rewards.append(list_)
    data_dict['Safe HAPPO--lagr-coef=0.75'] = episode_rewards

    episode_rewards = []
    for i in range(1):
        index = i + 1
        csv_path = "./mappo_lagr_ablation/manyant_" + config + "/mappo_lagrdot8_lagrCoef1e3/" + performance_type + "_seed" + str(
            index) + ".csv"
        list_ = read_csv_2_dict(csv_path)
        episode_rewards.append(list_)
    data_dict['Safe HAPPO--lagr-coef=0.8'] = episode_rewards

    episode_rewards = []
    for i in range(1):
        index = i + 1
        csv_path = "./mappo_lagr_ablation/manyant_" + config + "/mappo_lagrdot85_lagrCoef1e3/" + performance_type + "_seed" + str(
            index) + ".csv"
        list_ = read_csv_2_dict(csv_path)
        episode_rewards.append(list_)
    data_dict['Safe HAPPO--lagr-coef=0.85'] = episode_rewards

    episode_rewards = []
    for i in range(1):
        index = i + 1
        csv_path = "./mappo_lagr/manyagent_ant/manyant"+ config +"_lr9e5_exp1_3seeds/" + performance_type + "_seed" + str(index) + ".csv"
        list_ = read_csv_2_dict(csv_path)
        episode_rewards.append(list_)
    data_dict['Safe HAPPO--lagr-coef=0.9'] = episode_rewards

    episode_rewards = []
    for i in range(1):
        index = i + 1
        csv_path = "./mappo_lagr_ablation/manyant_" + config + "/mappo_lagrdot95_lagrCoef1e3/" + performance_type + "_seed" + str(
            index) + ".csv"
        list_ = read_csv_2_dict(csv_path)
        episode_rewards.append(list_)
    data_dict['Safe HAPPO--lagr-coef=0.95'] = episode_rewards

    episode_rewards = []
    for i in range(1):
        index = i + 1
        csv_path = "./mappo_lagr_ablation/manyant_" + config + "/mappo_lagr1_lagrCoef1e3/" + performance_type + "_seed" + str(
            index) + ".csv"
        list_ = read_csv_2_dict(csv_path)
        episode_rewards.append(list_)
    data_dict['Safe HAPPO--lagr-coef=1'] = episode_rewards
    """

    draw(data_dict, 0, smoothen_w)

    plt.title(scenario + " " + config, fontsize=18, pad=12)
    if need_legend:
        plt.legend(loc="upper left", fontsize=14)
    plt.grid()
    if need_legend:
        save_path = './Figure/ManyAgent_Ant/new_caption/' + scenario + config + performance_type + "_3seeds"+ '.pdf'
    else:
        save_path = './Figure/ManyAgent_Ant/new_caption/' + scenario + config + performance_type + "_3seeds"+ '.pdf'
    plt.savefig(save_path, format='pdf')
    plt.rcParams['figure.figsize'] = (12.0, 8.0)
    plt.show()

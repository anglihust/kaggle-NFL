import numpy as np
import pandas as pd
from xgboost import plot_importance
import xgboost as xgb
import datetime
from sklearn.preprocessing import StandardScaler
import os
from sklearn.model_selection import train_test_split
from matplotlib import pyplot


def data():
    def create_features(df, deploy=False):
        def map_weather(txt):
            ans = 1
            if pd.isna(txt):
                return 0
            if 'partly' in txt:
                ans *= 0.5
            if 'climate controlled' in txt or 'indoor' in txt:
                return ans * 3
            if 'sunny' in txt or 'sun' in txt:
                return ans * 2
            if 'clear' in txt:
                return ans
            if 'cloudy' in txt:
                return -ans
            if 'rain' in txt or 'rainy' in txt:
                return -2 * ans
            if 'snow' in txt:
                return -3 * ans
            return 0

        def strtofloat(x):
            try:
                return float(x)
            except:
                return -1

        def horiontal_speed(angle, play_direction, speed):
            if play_direction == 'left':
                radiance = np.mod((180 + angle) / 360 * 2 * np.pi, 2 * np.pi)
            else:
                radiance = np.mod(angle / 360 * 2 * np.pi, 2 * np.pi)
            h_h = np.abs(np.sin(radiance) * speed)
            return h_h

        def vertical_speed(angle, play_direction, speed):
            if play_direction == 'left':
                radiance = np.mod((180 + angle) / 360 * 2 * np.pi, 2 * np.pi)
            else:
                radiance = np.mod(angle / 360 * 2 * np.pi, 2 * np.pi)
            v_h = np.abs(np.cos(radiance) * speed)
            return v_h

        # normalize the direction
        def new_X(x_coordinate, play_direction):
            if play_direction == 'left':
                return 120.0 - x_coordinate
            else:
                return x_coordinate

        # standarize the line
        def new_line(rush_team, field_position, yardline):
            if rush_team == field_position:
                # offense starting at X = 0 plus the 10 yard endzone plus the line of scrimmage
                return 10.0 + yardline
            else:
                # half the field plus the yards between midfield and the line of scrimmage
                return 60.0 + (50 - yardline)

        # standarize the play angle
        def new_orientation(angle, play_direction):
            if play_direction == 'left':
                new_angle = 360.0 - angle
                if new_angle == 360.0:
                    new_angle = 0.0
                return new_angle
            else:
                return angle

        # calculate the distance
        def euclidean_distance(x1, y1, x2, y2):
            x_diff = (x1 - x2) ** 2
            y_diff = (y1 - y2) ** 2
            return np.sqrt(x_diff + y_diff)

        # runing back or not
        def back_direction(orientation):
            if orientation > 180.0:
                return 1
            else:
                return 0

        # update yardline in the original df file
        def update_yardline(df):
            new_yardline = df[df['NflId'] == df['NflIdRusher']]
            new_yardline['YardLine'] = new_yardline[['PossessionTeam', 'FieldPosition', 'YardLine']].apply(
                lambda x: new_line(x[0], x[1], x[2]), axis=1)
            new_yardline = new_yardline[['GameId', 'PlayId', 'YardLine']]

            return new_yardline

        # update orientation in the original df file
        def update_orientation(df, yardline):
            df['X'] = df[['X', 'PlayDirection']].apply(lambda x: new_X(x[0], x[1]), axis=1)
            df['Orientation'] = df[['Orientation', 'PlayDirection']].apply(lambda x: new_orientation(x[0], x[1]),
                                                                           axis=1)
            df['Dir'] = df[['Dir', 'PlayDirection']].apply(lambda x: new_orientation(x[0], x[1]), axis=1)

            df = df.drop('YardLine', axis=1)
            df = pd.merge(df, yardline, on=['GameId', 'PlayId'], how='inner')

            return df

        # features of going back?
        def back_features(df):
            carriers = df[df['NflId'] == df['NflIdRusher']][
                ['GameId', 'PlayId', 'NflIdRusher', 'X', 'Y', 'Orientation', 'Dir', 'YardLine', 'S', 'PlayDirection',
                 'TimeHandoff', 'PlayerBirthDate']]
            carriers['back_from_scrimmage'] = carriers['YardLine'] - carriers['X']
            carriers['back_oriented_down_field'] = carriers['Orientation'].apply(lambda x: back_direction(x))
            carriers['back_moving_down_field'] = carriers['Dir'].apply(lambda x: back_direction(x))
            carriers['horizontal_speed'] = carriers[['Dir', 'PlayDirection', 'S']].apply(
                lambda x: horiontal_speed(x[0], x[1], x[2]), axis=1)
            carriers['vertical_speed'] = carriers[['Dir', 'PlayDirection', 'S']].apply(
                lambda x: vertical_speed(x[0], x[1], x[2]), axis=1)
            carriers['TimeHandoff'] = carriers['TimeHandoff'].apply(
                lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
            carriers['PlayerBirthDate'] = carriers['PlayerBirthDate'].apply(
                lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))
            seconds_in_year = 60 * 60 * 24 * 365.25
            carriers['PlayerAge'] = carriers.apply(
                lambda row: (row['TimeHandoff'] - row['PlayerBirthDate']).total_seconds() / seconds_in_year, axis=1)

            carriers = carriers.rename(columns={'X': 'back_X',
                                                'Y': 'back_Y'})
            carriers = carriers[
                ['GameId', 'PlayId', 'NflIdRusher', 'back_X', 'back_Y', 'back_from_scrimmage',
                 'back_oriented_down_field',
                 'back_moving_down_field', 'PlayerAge']]  # 'vertical_speed','horizontal_speed']]

            return carriers

        def features_relative_to_back(df, carriers):
            player_distance = df[['GameId', 'PlayId', 'NflId', 'X', 'Y']]
            player_distance = pd.merge(player_distance, carriers, on=['GameId', 'PlayId'], how='inner')
            player_distance = player_distance[player_distance['NflId'] != player_distance['NflIdRusher']]
            player_distance['dist_to_back'] = player_distance[['X', 'Y', 'back_X', 'back_Y']].apply(
                lambda x: euclidean_distance(x[0], x[1], x[2], x[3]), axis=1)

            player_distance = player_distance.groupby(
                ['GameId', 'PlayId', 'back_from_scrimmage', 'back_oriented_down_field', 'back_moving_down_field']) \
                .agg({'dist_to_back': ['min', 'max', 'mean', 'std']}) \
                .reset_index()
            player_distance.columns = ['GameId', 'PlayId', 'back_from_scrimmage', 'back_oriented_down_field',
                                       'back_moving_down_field',
                                       'min_dist', 'max_dist', 'mean_dist', 'std_dist']

            return player_distance

        def defense_features(df):
            rusher = df[df['NflId'] == df['NflIdRusher']][['GameId', 'PlayId', 'Team', 'X', 'Y', 'S', 'PlayerHeight']]
            rusher.columns = ['GameId', 'PlayId', 'RusherTeam', 'RusherX', 'RusherY', 'RusherS', 'rush_PlayerHeight']

            defense = pd.merge(df, rusher, on=['GameId', 'PlayId'], how='inner')
            defense = defense[defense['Team'] != defense['RusherTeam']][
                ['GameId', 'PlayId', 'X', 'Y', 'RusherX', 'RusherY', 'S', 'RusherS', 'PlayerHeight', 'PlayerBirthDate',
                 'TimeHandoff']]

            t = defense['S'].replace(0.0, 2.31)
            defense['S'] = t.values
            t = defense['RusherS'].replace(0.0, 4.24)
            defense['RusherS'] = t.values

            defense['TimeHandoff'] = defense['TimeHandoff'].apply(
                lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
            defense['PlayerBirthDate'] = defense['PlayerBirthDate'].apply(
                lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))
            seconds_in_year = 60 * 60 * 24 * 365.25
            defense['PlayerAge'] = defense.apply(
                lambda row: (row['TimeHandoff'] - row['PlayerBirthDate']).total_seconds() / seconds_in_year, axis=1)
            defender_age_ave = defense.groupby('PlayId')['PlayerAge'].mean()

            defense['PlayerHeight_dense'] = defense['PlayerHeight'].apply(
                lambda x: 12 * int(x.split('-')[0]) + int(x.split('-')[1]))
            ave_player_height = defense.groupby('PlayId')['PlayerHeight_dense'].mean()
            defense['def_dist_to_back'] = defense[['X', 'Y', 'RusherX', 'RusherY']].apply(
                lambda x: euclidean_distance(x[0], x[1], x[2], x[3]), axis=1)
            defense['min_time_to_tackle'] = defense['def_dist_to_back'].div(defense['S'])
            min_time_to_tackle = defense.groupby('PlayId')['min_time_to_tackle'].min()
            ave_time_to_tackle = defense.groupby('PlayId')['min_time_to_tackle'].mean()
            defense['closest_defender_speed_ratio'] = defense['S'].div(defense['RusherS'])
            mean_speed_ratio = defense.groupby('PlayId')['closest_defender_speed_ratio'].mean()
            idx = defense.groupby('PlayId')['def_dist_to_back'].transform(min) == defense['def_dist_to_back']
            closest_defender_speed_ratio = defense[idx][['closest_defender_speed_ratio']]

            defense = defense.groupby(['GameId', 'PlayId']) \
                .agg({'def_dist_to_back': ['min', 'max', 'mean', 'std']}) \
                .reset_index()
            defense.columns = ['GameId', 'PlayId', 'def_min_dist', 'def_max_dist', 'def_mean_dist', 'def_std_dist']
            defense['min_time_to_tackle'] = min_time_to_tackle.values
            defense['ave_time_to_tackle'] = ave_time_to_tackle.values
            defense['mean_speed_ratio'] = mean_speed_ratio.values
            defense['closest_defender_speed_ratio'] = closest_defender_speed_ratio.values
            defense['defender_ave_height'] = ave_player_height.values
            defense['defender_age_ave'] = defender_age_ave.values
            return defense

        def static_features(df):
            df['WindSpeed'] = df['WindSpeed'].apply(
                lambda x: x.lower().replace('mph', '').strip() if not pd.isna(x) else x)
            df['WindSpeed'] = df['WindSpeed'].apply(
                lambda x: (int(x.split('-')[0]) + int(x.split('-')[1])) / 2 if not pd.isna(x) and '-' in x else x)
            df['WindSpeed'] = df['WindSpeed'].apply(
                lambda x: (int(x.split()[0]) + int(x.split()[-1])) / 2 if not pd.isna(x) and type(
                    x) != float and 'gusts up to' in x else x)
            df['WindSpeed'] = df['WindSpeed'].apply(strtofloat)

            static_features = df[df['NflId'] == df['NflIdRusher']][
                ['GameId', 'PlayId', 'X', 'Y', 'S', 'A', 'Dis', 'Orientation', 'Dir',
                 'YardLine', 'Quarter', 'Down', 'Distance', 'DefendersInTheBox', 'PlayerHeight', 'TimeHandoff',
                 'TimeSnap', 'WindSpeed', 'GameWeather', 'HomeScoreBeforePlay',
                 'VisitorScoreBeforePlay']].drop_duplicates()
            static_features['rush_height '] = static_features['PlayerHeight'].apply(
                lambda x: 12 * int(x.split('-')[0]) + int(x.split('-')[1]))
            static_features['TimeHandoff'] = static_features['TimeHandoff'].apply(
                lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
            static_features['TimeSnap'] = static_features['TimeSnap'].apply(
                lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
            static_features['TimeDelta'] = static_features.apply(
                lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)

            # static_features['WindSpeed_ob'] = static_features['WindSpeed'].apply(lambda x: x.lower().replace('mph', '').strip() if not pd.isna(x) else x)
            # static_features['WindSpeed_ob'] = static_features['WindSpeed_ob'].apply(lambda x: (int(x.split('-')[0])+int(x.split('-')[1]))/2 if not pd.isna(x) and '-' in x else x)
            # static_features['WindSpeed_ob'] = static_features['WindSpeed_ob'].apply(lambda x: (int(x.split()[0])+int(x.split()[-1]))/2 if not pd.isna(x) and type(x)!=float and 'gusts up to' in x else x)
            # static_features['WindSpeed_dense'] = static_features['WindSpeed_ob'].apply(strtofloat)

            static_features['GameWeather_process'] = static_features['GameWeather'].str.lower()
            static_features['GameWeather_process'] = static_features['GameWeather_process'].apply(
                lambda x: "indoor" if not pd.isna(x) and "indoor" in x else x)
            static_features['GameWeather_process'] = static_features['GameWeather_process'].apply(
                lambda x: x.replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party',
                                                                                            'partly') if not pd.isna(
                    x) else x)
            static_features['GameWeather_process'] = static_features['GameWeather_process'].apply(
                lambda x: x.replace('clear and sunny', 'sunny and clear') if not pd.isna(x) else x)
            static_features['GameWeather_process'] = static_features['GameWeather_process'].apply(
                lambda x: x.replace('skies', '').replace("mostly", "").strip() if not pd.isna(x) else x)
            static_features['GameWeather_dense'] = static_features['GameWeather_process'].apply(map_weather)

            static_features['diffScoreBeforePlay'] = static_features['HomeScoreBeforePlay'] - static_features[
                'VisitorScoreBeforePlay']

            static_features['DefendersInTheBox'] = static_features['DefendersInTheBox'].fillna(
                np.mean(static_features['DefendersInTheBox']))

            static_features = static_features[
                ['GameId', 'PlayId', 'X', 'Y', 'S', 'A', 'Dis', 'Orientation', 'Dir', 'YardLine', 'Quarter', 'Down',
                 'Distance', 'DefendersInTheBox', 'rush_height ', 'TimeDelta', 'GameWeather_dense',
                 'diffScoreBeforePlay']]
            return static_features

        def split_personnel(s):
            splits = s.split(',')
            for i in range(len(splits)):
                splits[i] = splits[i].strip()

            return splits

        def defense_formation(l):
            dl = 0
            lb = 0
            db = 0
            other = 0

            for position in l:
                sub_string = position.split(' ')
                if sub_string[1] == 'DL':
                    dl += int(sub_string[0])
                elif sub_string[1] in ['LB', 'OL']:
                    lb += int(sub_string[0])
                else:
                    db += int(sub_string[0])

            counts = (dl, lb, db, other)

            return counts

        def offense_formation(l):
            qb = 0
            rb = 0
            wr = 0
            te = 0
            ol = 0

            sub_total = 0
            qb_listed = False
            for position in l:
                sub_string = position.split(' ')
                pos = sub_string[1]
                cnt = int(sub_string[0])

                if pos == 'QB':
                    qb += cnt
                    sub_total += cnt
                    qb_listed = True
                # Assuming LB is a line backer lined up as full back
                elif pos in ['RB', 'LB']:
                    rb += cnt
                    sub_total += cnt
                # Assuming DB is a defensive back and lined up as WR
                elif pos in ['WR', 'DB']:
                    wr += cnt
                    sub_total += cnt
                elif pos == 'TE':
                    te += cnt
                    sub_total += cnt
                # Assuming DL is a defensive lineman lined up as an additional line man
                else:
                    ol += cnt
                    sub_total += cnt

            # If not all 11 players were noted at given positions we need to make some assumptions
            # I will assume if a QB is not listed then there was 1 QB on the play
            # If a QB is listed then I'm going to assume the rest of the positions are at OL
            # This might be flawed but it looks like RB, TE and WR are always listed in the personnel
            if sub_total < 11:
                diff = 11 - sub_total
                if not qb_listed:
                    qb += 1
                    diff -= 1
                ol += diff

            counts = (qb, rb, wr, te, ol)

            return counts

        def personnel_features(df):
            personnel = df[['GameId', 'PlayId', 'OffensePersonnel', 'DefensePersonnel']].drop_duplicates()
            personnel['DefensePersonnel'] = personnel['DefensePersonnel'].apply(lambda x: split_personnel(x))
            personnel['DefensePersonnel'] = personnel['DefensePersonnel'].apply(lambda x: defense_formation(x))
            personnel['num_DL'] = personnel['DefensePersonnel'].apply(lambda x: x[0])
            personnel['num_LB'] = personnel['DefensePersonnel'].apply(lambda x: x[1])
            personnel['num_DB'] = personnel['DefensePersonnel'].apply(lambda x: x[2])

            personnel['OffensePersonnel'] = personnel['OffensePersonnel'].apply(lambda x: split_personnel(x))
            personnel['OffensePersonnel'] = personnel['OffensePersonnel'].apply(lambda x: offense_formation(x))
            personnel['num_QB'] = personnel['OffensePersonnel'].apply(lambda x: x[0])
            personnel['num_RB'] = personnel['OffensePersonnel'].apply(lambda x: x[1])
            personnel['num_WR'] = personnel['OffensePersonnel'].apply(lambda x: x[2])
            personnel['num_TE'] = personnel['OffensePersonnel'].apply(lambda x: x[3])
            personnel['num_OL'] = personnel['OffensePersonnel'].apply(lambda x: x[4])

            # Let's create some features to specify if the OL is covered
            personnel['OL_diff'] = personnel['num_OL'] - personnel['num_DL']
            personnel['OL_TE_diff'] = (personnel['num_OL'] + personnel['num_TE']) - personnel['num_DL']
            # Let's create a feature to specify if the defense is preventing the run
            # Let's just assume 7 or more DL and LB is run prevention
            personnel['run_def'] = (personnel['num_DL'] + personnel['num_LB'] > 6).astype(int)

            personnel.drop(['OffensePersonnel', 'DefensePersonnel'], axis=1, inplace=True)

            return personnel

        def combine_features(relative_to_back, defense, static, personnel, deploy=deploy):
            df = pd.merge(relative_to_back, defense, on=['GameId', 'PlayId'], how='inner')
            df = pd.merge(df, static, on=['GameId', 'PlayId'], how='inner')
            df = pd.merge(df, personnel, on=['GameId', 'PlayId'], how='inner')

            if not deploy:
                df = pd.merge(df, outcomes, on=['GameId', 'PlayId'], how='inner')

            return df

        outcomes = df[['GameId', 'PlayId', 'Yards']].drop_duplicates()
        yardline = update_yardline(df)
        df = update_orientation(df, yardline)
        back_feats = back_features(df)
        rel_back = features_relative_to_back(df, back_feats)
        def_feats = defense_features(df)
        static_feats = static_features(df)
        personnel = personnel_features(df)
        basetable = combine_features(rel_back, def_feats, static_feats, personnel, deploy=deploy)
        return basetable

    def process_two(t_):
        t_['fe1'] = pd.Series(np.sqrt(np.absolute(np.square(t_.X.values) - np.square(t_.Y.values))))
        t_['fe5'] = np.square(t_['S'].values) + 2 * t_['A'].values * t_['Dis'].values  # N
        t_['fe7'] = np.arccos(np.clip(t_['X'].values / t_['Y'].values, -1, 1))  # N
        t_['fe8'] = t_['S'].values / np.clip(t_['fe1'].values, 0.6, None)
        radian_angle = (90 - t_['Dir']) * np.pi / 180.0
        t_['fe10'] = np.abs(t_['S'] * np.cos(radian_angle))
        t_['fe11'] = np.abs(t_['S'] * np.sin(radian_angle))
        return t_

    read_path = os.path.join(os.getcwd(), 'train.csv')
    train = pd.read_csv(read_path, dtype={'WindSpeed': 'object'})
    train_basetable = create_features(train, False)
    X = train_basetable.copy()
    yards = X.Yards
    y = np.zeros((yards.shape[0], 199))
    for idx, target in enumerate(list(yards)):
        y[idx][99 + target] = 1
    X = process_two(X)
    X.drop(['GameId', 'PlayId', 'Yards'], axis=1, inplace=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    return x_train, y_train, x_test, y_test, X, yards,y

def feature_importance(x_train, y_train):
    model = xgb.XGBRegressor()
    # model = xgb.XGBRegressor(learning_rate = 0.05, n_estimators=300, max_depth=5)
    model.fit(x_train, y_train)
    plot_importance(model)
    pyplot.show()

def  feature_importance_selection(x_train, y_train, x_test, y_test,X,yards,y):
    number = (
    48, 20, 3, 13, 11, 0, 23, 9, 17, 14, 22, 24, 45, 7, 49, 19, 12, 10, 6, 5, 46, 4, 44, 32, 35, 28, 18, 21, 27, 47, 16,
    15, 29, 8)
    x_train=x_train[:,number]
    x_test=x_test[:,number]
    X = X[:, number]
    return x_train, y_train, x_test, y_test, X, yards,y
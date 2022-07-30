from flask import Flask, render_template
from forms import ReasonForm
import os
import pandas as pd
import json
from datetime import datetime
from pytz import timezone
from config import SECRET_KEY

## Run initialization code "cnn_ui_reason_prep.py" first


#*##########

verify_code = '######'

#*##########


app = Flask(__name__)

app.config['SECRET_KEY'] = SECRET_KEY
csvname_curr = 'current.csv'
csvname_res = 'results.csv'
tz = timezone('EST')

def unshuffle(results, rnd_order):
        results_2 = [0] * 4
        k = 0
        for j in rnd_order[1:]:
            results_2[j-1] = results[k]
            k += 1
        return results_2


@app.route("/", methods=['GET', 'POST'])
def index():
    form = ReasonForm()
    result_path = 'output/%s' % csvname_res
    try:
        df_results = pd.read_csv(result_path, index_col=0)
        df_current = pd.read_csv('output/%s' % csvname_curr, index_col=0)
        idx_list = df_results['img_idx']
        current_order = int(df_current['current_order'])
        current_idx = idx_list[current_order].split('.jpg')[0]
        finish_text = ""
        results = []
        results_all= []
        rnd_order = json.loads(df_results.loc[current_order, 'rnd_order'])
        form_display = True

        if form.validate_on_submit():
            results.append(form.reason_input_1.data)
            results.append(form.reason_input_2.data)
            results.append(form.reason_input_3.data)
            results.append(form.reason_input_4.data)
            
            results_all += unshuffle(results, rnd_order)

            results = []
            results.append(form.reason_input_5.data)
            results.append(form.reason_input_6.data)
            results.append(form.reason_input_7.data)
            results.append(form.reason_input_8.data)

            results_all += unshuffle(results, rnd_order)

            results = []
            results.append(form.reason_input_9.data)
            results.append(form.reason_input_10.data)
            results.append(form.reason_input_11.data)
            results.append(form.reason_input_12.data)

            results_all += unshuffle(results, rnd_order)

            df_results.loc[current_order, :] = [current_idx] + results_all + [str(rnd_order)]
            
            if current_order + 1 < len(idx_list):
                current_order += 1
                current_idx = idx_list[current_order].split('.jpg')[0]
                rnd_order = json.loads(df_results.loc[current_order, 'rnd_order'])

            else:
                finish_text = f"Thank You! Your verification code is: {verify_code} (only appears once)"
                form_display = False
                os.remove(result_path)
                timestr = datetime.now(tz).strftime("%Y%m%d_%H%M%S_%f")[:-4]
                result_path = f"output/done/results_{timestr}.csv"
                pass

            df_results.to_csv(result_path)

        else:
            pass    
        
        #### Radio default selections

        # form.reason_input_1.data = None
        # form.reason_input_2.data = None
        # form.reason_input_3.data = None
        # form.reason_input_4.data = None
        # form.reason_input_5.data = None
        # form.reason_input_6.data = None
        # form.reason_input_7.data = None
        # form.reason_input_8.data = None

        form.reason_input_1.data = 'True'
        form.reason_input_2.data = 'True'
        form.reason_input_3.data = 'True'
        form.reason_input_4.data = 'True'
        form.reason_input_5.data = 'True'
        form.reason_input_6.data = 'True'
        form.reason_input_7.data = 'True'
        form.reason_input_8.data = 'True'
        
        # form.reason_input_9.data = None
        # form.reason_input_10.data = None
        # form.reason_input_11.data = None
        # form.reason_input_12.data = None

        form.reason_input_9.data = '5'
        form.reason_input_10.data = '5'
        form.reason_input_11.data = '5'
        form.reason_input_12.data = '5'

        img_paths = []

        for i in rnd_order:
            img_paths.append(f'static/images/{i}/{current_idx}.jpg')

        df_current.loc[0, 'current_order'] = current_order
        df_current.loc[0, 'current_idx'] = idx_list[current_order]

        df_current.to_csv('output/%s' % csvname_curr)

        q1_q2_display = False
        q1_q2_display = True

        if q1_q2_display == False:
            q1_q2_display = "display: none;"
        else:
            q1_q2_display = ""

        return render_template('index.html', title='Main', current_order=str(current_order+1), len_idx = len(idx_list), finish_text=finish_text, img_paths=img_paths, form=form, form_display=form_display, q1_q2_display=q1_q2_display)

    except:
        return render_template('blank.html', title='Blank')

if __name__ == '__main__':
    app.run(debug=True)

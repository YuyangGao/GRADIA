from flask_wtf import FlaskForm
from wtforms import SubmitField, RadioField 

class ReasonForm(FlaskForm):
    reason_input_1 = RadioField('Reasonable or not', choices=[(True,'yes'), (False,'no')])
    reason_input_2 = RadioField('Reasonable or not', choices=[(True,'yes'), (False,'no')])
    reason_input_3 = RadioField('Reasonable or not', choices=[(True,'yes'), (False,'no')])
    reason_input_4 = RadioField('Reasonable or not', choices=[(True,'yes'), (False,'no')])
    reason_input_5 = RadioField('Reasonable or not', choices=[(True,'yes'), (False,'no')])
    reason_input_6 = RadioField('Reasonable or not', choices=[(True,'yes'), (False,'no')])
    reason_input_7 = RadioField('Reasonable or not', choices=[(True,'yes'), (False,'no')])
    reason_input_8 = RadioField('Reasonable or not', choices=[(True,'yes'), (False,'no')])
    
    reason_input_9 = RadioField('Rating', choices=[
                                                    (5,'5 (Excellent)'), 
                                                    (4,'4 (Above Average)'), 
                                                    (3,'3 (Average)'), 
                                                    (2,'2 (Below Average)'), 
                                                    (1,'1 (Very Poor)')
                                                    ])
    reason_input_10 = RadioField('Rating', choices=[
                                                    (5,'5 (Excellent)'), 
                                                    (4,'4 (Above Average)'), 
                                                    (3,'3 (Average)'), 
                                                    (2,'2 (Below Average)'), 
                                                    (1,'1 (Very Poor)')
                                                    ])
    reason_input_11 = RadioField('Rating', choices=[
                                                    (5,'5 (Excellent)'), 
                                                    (4,'4 (Above Average)'), 
                                                    (3,'3 (Average)'), 
                                                    (2,'2 (Below Average)'), 
                                                    (1,'1 (Very Poor)')
                                                    ])
    reason_input_12 = RadioField('Rating', choices=[
                                                    (5,'5 (Excellent)'), 
                                                    (4,'4 (Above Average)'), 
                                                    (3,'3 (Average)'), 
                                                    (2,'2 (Below Average)'), 
                                                    (1,'1 (Very Poor)')
                                                    ])




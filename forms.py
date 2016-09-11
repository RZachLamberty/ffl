from flask_wtf import Form
from wtforms import SelectField, SubmitField
from wtforms.validators import DataRequired


class DraftForm(Form):
    playername = SelectField('Player Name', validators=[DataRequired()])
    draftteam = SelectField('Drafting Team', validators=[DataRequired()])
    submit = SubmitField('Draft')


def update_draftform_choices(draftform, ffldraft):
    draftform.playername.choices = sorted(
        zip(ffldraft.undrafted.index, ffldraft.undrafted.playername),
        key=lambda row: row[1]
    )
    draftform.draftteam.choices = list(zip(
        ffldraft.leagueteams.index,
        [
            '{} ({})'.format(n, o)
            for (n, o) in ffldraft.leagueteams[['name', 'owner']].values
        ]
    ))

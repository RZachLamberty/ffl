/* --------------------------------------------------------------------------
 * datatable render functions
 * -------------------------------------------------------------------------- */

function undo_form(data, type, row, meta) {
    return '<div class="undoDraftFormContainer"> \
        <form class="undoDraftForm" action="' + window.undourl + '" method="POST"> \
            <input type="hidden" id="player_index" name="player_index" value="' + row.player_index + '"> \
            <div class="text-center"> \
                <button class="btn btn-danger btn-sm"> \
                    <span class="glyphicon glyphicon-remove centericon cursoricon" aria-hidden="true"></span> \
                </button> \
            </div> \
        </form> \
    </div>';
}

/* --------------------------------------------------------------------------
 * making tables
 * -------------------------------------------------------------------------- */

function draw_player_data_table(tabId, playerData) {
    $(document).ready(function() {
        var playertable = $(tabId).DataTable({
            'data': playerData,
            'columns': [
                { 'title': 'Player', 'data': 'playername' },
                { 'title': 'NFL Team', 'data': 'team' },
                { 'title': 'Draft Status', 'data': 'status_type' },
                { 'title': 'Position', 'data': 'pos' },
                { 'title': 'Points (total)', 'data': 'pts_total' }
            ],
            'order': [4, 'desc'],
            'pageLength': 50
        });
    });
}

function draw_draft_history_table(tabId, histData) {
    $(document).ready(function() {
        var histtable = $(tabId).DataTable({
            'data': histData,
            'columns': [
                { 'title': 'Draft Number', 'data': 'draftnum' },
                { 'title': 'Player', 'data': 'player_name' },
                { 'title': 'Team', 'data': 'team_name' },
                { 'title': 'Undo', 'render': undo_form }
            ],
            'order': [0, 'desc'],
            'pageLength': 50
        });
    });
}


/* --------------------------------------------------------------------------
 * plotly replacement value plot
 * -------------------------------------------------------------------------- */

function replacement_plot(plotdiv, replacementPlotData) {
    $(document).ready(function() {
        var layout = {
            title: 'Replacement Value',
            xaxis: {
                title: 'position depth'
            },
            yaxis: {
                title: 'replacement points'
            }
        };

        Plotly.newPlot(plotdiv, replacementPlotData, layout)
    });
}

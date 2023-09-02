import Vue from 'vue'
import Vuex from 'vuex'
import axios from 'axios'
import Highlight from '@/plugins/highlight.js'
import ColorAssigner from '@/plugins/color.js'
import { adjustHexColorOpacity } from '@/plugins/utils.js'
import { schemeTableau10 } from 'd3-scale-chromatic'
import { scaleOrdinal } from 'd3-scale'
import { Shape, Color, get_polarity_color } from '@/plugins/utils.js';
import * as d3 from 'd3'
import { index, select, selectAll } from 'd3'

axios.defaults.headers.common['Access-Control-Allow-Origin'] = '*';
axios.defaults.headers.common['Access-Control-Allow-Methods'] = 'GET, POST, PATCH, PUT, DELETE, OPTIONS';
//mount Vuex
Vue.use(Vuex);

//create VueX
const store = new Vuex.Store({
    state:{ // data
        url: '/api',
        modelurl: '/modelapi',
        color_scale: scaleOrdinal(schemeTableau10),
        samples: [],
        all_samples: [],
        row_index: [],
        col_index: [],

        meta_data: [],
        label_names: [],

        num_row_cluster: 0,
        num_col_cluster: 0,
        R: null,
        old_R: null,
        lam: null,
        lam_: null,

        constraint: null,
        opt_result: null,
        matrix: null,

        pcp_attributes: [],

        color_type: 'groundtruth',
        color_opacity: 'uniform',
        colors: [],
        scatter_colors: [],
        opacity: [],
        line_attribute: 'none',
        
        highlight: new Highlight(),
        high_pos_validation: new Highlight(),
        high_neg_validation: new Highlight(),
        highlight_row_index: [],
        highlight_col_index: [],
        color_assigner: new ColorAssigner(),
        legend: [],
        color_map_list: [],

        update_cluster_by_idx: -1,
        redraw_cnt: 0,
        lam_type: 1, // 1 means current, 0 means previous, -1 means diff

        showgt: false,
        consider_lam: true,
        case_step: 0,

        diff_threshold: 0.1,
    },
    mutations:{ // function to modify the state
        init_data(state, data) {
            window.d3 = d3;
            window.state = state;
            console.log('init_data');
            state.meta_data = data.meta_data;
            state.label_names = state.meta_data.label_names.en;
            state.full_col_cluster = data.full_col_cluster;
            state.color_assigner.init(state.label_names);
            state.legend = state.color_assigner.legend;
            state.color_map_list = state.color_assigner.color_map_list;

            state.all_samples = Array(data.meta_data.num_samples).fill(null);
            let compressImage = function(d, base64string, quality = 0.2) {
                let canvas = document.createElement("canvas");
                canvas.width = 30;
                canvas.height = 30;
                let ctx = canvas.getContext("2d");  
                let newImage = new Image();
                newImage.src = "data:image/png;base64," + base64string;
                newImage.onload = function() {
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.drawImage(newImage, 0, 0, canvas.width, canvas.height);
                    let newBase64 = canvas.toDataURL("image/png", quality);
                    d.compressed_image = newBase64;
                }
            }
            data.samples.forEach((d) => {
                d.idx = d.index;
                d.color =  Color.WHITE_POINT;
                d.label_color = state.color_map_list[d.label];
                d.shape = Shape.CIRCLE;
                d.large_shape = Shape.LARGE_CIRCLE;
                d.previous_label = d.label;
                d.diff = false;
                compressImage(d, d.image);
                state.all_samples[d.index] = d;
            });

            // Cluster
            state.row_index = data.cluster.row_index;
            state.row_index.forEach(idx => {
                state.all_samples[idx].color = Color.darkgrey;
                state.all_samples[idx].shape = Shape.SQUARE;
                state.all_samples[idx].large_shape = Shape.LARGE_SQUARE;
            });
            state.col_index = data.cluster.col_index;
            state.display_idx = data.cluster.display_idx;
            state.display_index = state.display_idx.map(i => state.col_index[i]);
            data.cluster.row_index.forEach((idx, j) => {
                state.all_samples[idx].isVal = true;
                state.all_samples[idx].row_i = j;
                state.all_samples[idx].row_cluster = data.cluster.row_cluster_label[j];
                state.all_samples[idx].row_sub_cluster_label = data.cluster.row_sub_cluster_label[j];
                state.all_samples[idx].row_x = data.cluster.row_x[j];
            });
            data.cluster.col_index.forEach((idx, j) => {
                state.all_samples[idx].isTrain = true;
                state.all_samples[idx].col_j = j;
                state.all_samples[idx].col_cluster = data.cluster.col_cluster_label[j];
                state.all_samples[idx].col_sub_cluster_label = data.cluster.col_sub_cluster_label[j];
                state.all_samples[idx].col_x = data.cluster.col_x[j];
            });
            state.num_row_cluster = data.cluster.num_row_cluster;
            state.num_col_cluster = data.cluster.num_col_cluster;
            state.R = data.cluster.R;

            // constraint
            state.constraint = data.constraint;
            state.constraint.pos_idx.forEach((i) => {
                state.all_samples[state.col_index[i]].is_pos = true;
                state.all_samples[state.col_index[i]].color = Color.GREEN_POINT;
            });
            state.constraint.neg_idx.forEach((i) => {
                state.all_samples[state.col_index[i]].is_neg = true;
                state.all_samples[state.col_index[i]].color = Color.RED_POINT;
            });
            // opt_result
            this.commit("set_opt_result", data.opt_result);
        },
        add_highlight_idx(state, {highlight, source, idx}){
            highlight.add_highlight(source, idx);
            state.highlight_row_index = highlight.get_row_index();
            state.highlight_col_index = highlight.get_col_index();
        },
        rm_highlight_idx(state, {highlight, source}){
            highlight.rm_highlight(source);
            state.highlight_row_index = highlight.get_row_index();
            state.highlight_col_index = highlight.get_col_index();
        },
        set_matrix(state, matrix){
            state.matrix = matrix;
        },
        redraw_all(state) {
            state.redraw_cnt += 1;
        },
        set_update_cluster_by_idx(state, idx) {
            state.update_cluster_by_idx = idx;
        },
        update_color(state, labels){
            state.color_assigner.update(labels);
            state.legend = state.color_assigner.legend;
            state.color_map_list = state.color_assigner.color_map_list;
        },
        set_opt_result(state, opt_result) {
            let old_R = state.old_R ? state.old_R : state.R;
            state.old_R = null;
            let old_lam_sum = opt_result.old_lam.sum();
            let new_lam_sum = opt_result.new_lam.sum();
            state.opt_result = opt_result
            state.lam_ = [opt_result.old_lam.map(val => val / old_lam_sum), opt_result.new_lam.map(val => val / new_lam_sum)];
            
            state.row_index.forEach((idx, i) => {
                let sample = state.all_samples[idx];
                sample.lam_ = [state.lam_[0][i], state.lam_[1][i]];
                let lam = state.lam_[1][i];
                sample.influence_list = state.col_index.map((col_idx, j) => lam * state.R[i][j]);
                sample.sorted_influence_list = state.col_index.map((col_idx, j) => ({val: lam * state.R[i][j], col_idx: col_idx}))
                                    .sort((a,b) => a.val-b.val).map(d => d.col_idx);
                sample.raw_influence_list = state.col_index.map((col_idx, j) => state.R[i][j]);
                sample.sorted_raw_influence_list = state.col_index.map((col_idx, j) => ({val: state.R[i][j], col_idx: col_idx}))
                                    .sort((a,b) => a.val-b.val).map(d => d.col_idx);
            });
            state.col_index.forEach((idx, j) => {
                let old_score_list = state.lam_[0].map((l, i) => l * old_R[i][j]);
                let new_score_list = state.lam_[1].map((l, i) => l * state.R[i][j]);
                let sample = state.all_samples[idx];
                sample.score_list_ = [old_score_list, new_score_list];
                sample.score_ = [old_score_list.sum(), new_score_list.sum()];
                let old_sorted_score_list = old_score_list.map((val,i) => ({val:val, row_idx: state.row_index[i]}))
                    .sort((a, b) => a.val-b.val).map(d => d.row_idx);
                let new_sorted_score_list = new_score_list.map((val,i) => ({val:val, row_idx: state.row_index[i]}))
                    .sort((a, b) => a.val-b.val).map(d => d.row_idx);
                sample.sorted_score_list_ = [old_sorted_score_list, new_sorted_score_list];
                sample.raw_score_list = state.lam_[0].map((l, i) => state.R[i][j]);
                sample.raw_score = sample.raw_score_list.sum();
                sample.sorted_raw_score_list = sample.raw_score_list.map((val,i) => ({val:val, row_idx: state.row_index[i]}))
                .sort((a, b) => a.val-b.val).map(d => d.row_idx);
            });
            const old_pos_score_sum = state.col_index.map(idx => state.all_samples[idx].score_[0]).filter(d => d>0).sum();
            const new_pos_score_sum = state.col_index.map(idx => state.all_samples[idx].score_[1]).filter(d => d>0).sum();
            state.col_index.forEach(idx => {
                state.all_samples[idx].weight_ = [state.all_samples[idx].score_[0]>0 ? state.all_samples[idx].score_[0] / old_pos_score_sum : 0, 
                                                  state.all_samples[idx].score_[1]>0 ? state.all_samples[idx].score_[1] / new_pos_score_sum : 0]
            });
            const old_pos_mean = state.col_index.map(idx => state.all_samples[idx].score_[0]).filter(d => d>0).mean();
            const old_neg_mean = -state.col_index.map(idx => state.all_samples[idx].score_[0]).filter(d => d<0).mean();
            const new_pos_mean = state.col_index.map(idx => state.all_samples[idx].score_[1]).filter(d => d>0).mean();
            const new_neg_mean = -state.col_index.map(idx => state.all_samples[idx].score_[1]).filter(d => d<0).mean();
            state.col_index.forEach(idx => {
                let sample = state.all_samples[idx];
                // sample.shape = (sample.score > 0 && sample.is_neg || sample.score < 0 && sample.is_pos) ? Shape.CROSS : Shape.CIRCLE;
                sample.color_value_ = [sample.score_[0]>0 ? sample.score_[0] / old_pos_mean : sample.score_[0] / old_neg_mean, 
                                        sample.score_[1]>0 ? sample.score_[1] / new_pos_mean : state.all_samples[idx].score_[1] / new_neg_mean]
            });
            state.all_samples.forEach((sample, index) => {
                sample.full_score_ = [opt_result.old_full_weight[index],opt_result.new_full_weight[index]]
            })
            this.commit("choose_lam");
            this.commit("set_threshold")
            this.commit("redraw_all");
            this.commit("stat");
        },
        update_R(state, {data}) {
            console.log('commit update R', data);


            // Cluster
            state.row_index = data.row_index;
            state.col_index = data.col_index;
            data.add_row_index.forEach(idx => {
                state.all_samples[idx].color = Color.DARK_GREEN;
                state.all_samples[idx].shape = Shape.SQUARE;
                state.all_samples[idx].large_shape = Shape.LARGE_SQUARE;
                state.all_samples[idx].isVal = true;
                state.all_samples[idx].row_i = state.row_index.indexOf(idx);
                // state.all_samples[idx].row_cluster = data.cluster.row_cluster_label[j];
                // state.all_samples[idx].row_x = data.cluster.row_x[j];

                state.all_samples[idx].col_j = undefined;
                state.all_samples[idx].col_cluster = undefined
                state.all_samples[idx].col_sub_cluster_label = undefined;
                state.all_samples[idx].col_x = undefined;
            });

            for (let row of data.rows) {
                let cluster = data.cluster_info[row];
                let i = 0
                for (let idx of cluster.index) {
                    state.all_samples[idx].row_x = cluster.row_x[i];
                    state.all_samples[idx].row_cluster = cluster.cluster_label;
                    i += 1;
                }
            }

            state.old_R = null;
            state.R = data.R;

            state.constraint = data.constraint;
            state.constraint.pos_idx.forEach((i) => {
                state.all_samples[state.col_index[i]].is_pos = true;
                // state.all_samples[state.col_index[i]].color = Color.GREEN_POINT;
            });
            state.constraint.neg_idx.forEach((i) => {
                state.all_samples[state.col_index[i]].is_neg = true;
                // state.all_samples[state.col_index[i]].color = Color.RED_POINT;
            });

            state.constraint.up_idx.forEach((i, local_idx) => {
                let sample = state.all_samples[state.col_index[i]];
                sample.set_score = state.constraint.up_target[local_idx];
            });
            state.constraint.down_idx.forEach((i, local_idx) => {
                let sample = state.all_samples[state.col_index[i]];
                sample.set_score = state.constraint.down_target[local_idx];
            });
            // this.commit("set_opt_result", data.opt_result);
        },

        replace_R(state, {data}) {
            console.log('commit replace R', data);
            state.old_R = state.R;
            state.R = data.R;
            for (let row of data.rows) {
                let cluster = data.cluster_info[row];
                let i = 0
                for (let idx of cluster.index) {
                    state.all_samples[idx].row_x = cluster.row_x[i];
                    state.all_samples[idx].row_cluster = cluster.cluster_label;
                    // state.all_samples[idx].row_sub_cluster_label = cluster.sub_cluster_label[i];
                    i += 1;
                }
            }
            // this.commit("set_opt_result", state.opt_result);
        },

        set_lam_type(state, lam_type) {
            state.lam_type = lam_type;
            this.commit("choose_lam");
        },
        choose_lam(state) {
            let lam_type = state.lam_type;
            if (lam_type === -1) lam_type = 1;
            state.lam = state.lam_[lam_type];
            state.row_index.forEach((idx, i) => {
                let sample = state.all_samples[idx];
                sample.lam = sample.lam_[lam_type];
            });
            state.col_index.forEach((idx, j) => {
                let sample = state.all_samples[idx];
                sample.score_list = sample.score_list_[lam_type]
                sample.score = sample.score_[lam_type]
                sample.sorted_score_list = sample.sorted_score_list_[lam_type];
                sample.weight = sample.weight_[lam_type];
                // sample.inconsistent = (sample.score > 0 && sample.is_neg || sample.score < 0 && sample.is_pos);
                sample.inconsistent = (sample.score > 1e-4 && sample.is_neg || sample.score < -1e-4 && sample.is_pos);
                sample.shape = sample.inconsistent ? Shape.CROSS : Shape.CIRCLE;
                sample.color_value = sample.color_value_[lam_type];
                sample.color = get_polarity_color(sample.color_value);
            });
            state.all_samples.forEach((sample) => {
                sample.full_score = sample.full_score_[lam_type];
            });
        },
        async set_gamma(state, gamma){
            const resp = await axios.post(`${state.url}/set_gamma`,{
                'gamma': gamma,
            },{'Access-Control-Allow-Origin': '*'});
        },
        set_threshold(state, threshold){
            console.log('set threshold', threshold);
            if (threshold) state.diff_threshold = threshold;
            let col_samples = state.col_index.map(idx => state.all_samples[idx])
            let diff_values = col_samples.map(d => Math.abs(d.score_[1] - d.score_[0]));
            let value_thres = diff_values.quantile(1-state.diff_threshold);
            col_samples.forEach(d => d.diff = Math.abs(d.score_[1] - d.score_[0]) > value_thres);
        },
        make_corrections(state, {row_idx, col_idx, up_idx, down_idx}) {
            col_idx.forEach(idx => {
                let sample = state.all_samples[idx];
                sample.is_pos = sample.correct;
                sample.is_neg = !sample.correct;
                sample.color = sample.is_pos ? Color.GREEN_POINT : Color.RED_POINT;
                sample.corrected = true;
            })
            row_idx.forEach(idx => {
                let sample = state.all_samples[idx];
                sample.set_lam = 0;
                sample.corrected = true;
            })
            up_idx.forEach(idx => {
                if (!state.constraint.up_idx.includes(idx)) {
                    let sample = state.all_samples[idx];
                    sample.set_score = -sample.score;
                }
            })
            down_idx.forEach(idx => {
                if (!state.constraint.down_idx.includes(idx)) {
                    let sample = state.all_samples[idx];
                    sample.set_score = -sample.score;
                }
            })
        },
        async make_easy_corrections(state, {remove_reward, add_reward, replace_reward, reduce_reward, enlarge_reward, add_pos_constraint, remove_constraint, add_neg_constraint, add_up, add_down}) {
            if (remove_reward) {
                for (let reward_list of remove_reward) {
                    reward_list.forEach(idx => {
                        let sample = state.all_samples[idx];
                        state.constraint.lam_upper[sample.row_i] = 0
                        // sample.corrected = true;
                    })
                }
            }
            if (add_reward) {
                let row_index = [];
                let row_label = [];
                for (let reward_list of add_reward) {
                    row_index = [...row_index, ...reward_list[0]];
                    row_label = [...row_label, ...reward_list[1]];
                    reward_list[0].forEach(idx => {
                        let sample = state.all_samples[idx];
                        state.constraint.lam_lower[sample.row_i] = 1 / state.row_index.length / 2;
                    })
                    let data = await store.dispatch('fetch_relationship', {'row_index': row_index, 'row_label': row_label})
                }
            }
            if (replace_reward) {
                let row_index = [];
                let row_label = [];
                for (let reward_list of replace_reward) {
                    row_index = [...row_index, ...reward_list[0]];
                    row_label = [...row_label, ...reward_list[1]];
                    reward_list[0].forEach(idx => {
                        let sample = state.all_samples[idx];
                        // state.constraint.lam_lower[sample.row_i] = 1 / state.row_index.length - 0.001;
                        state.constraint.lam_lower[sample.row_i] = 1 / state.row_index.length / 2;
                    })
                }
                let data = await store.dispatch('update_existing_relationship', {'row_index': row_index, 'row_label': row_label})
            }        
            if (remove_constraint) {
                for (let constraint_list of remove_constraint) {
                    constraint_list.forEach(idx => {
                        let sample = state.all_samples[idx];
                        sample.is_pos = false;
                        sample.is_neg = false;
                        sample.color =  Color.WHITE_POINT;
                        // sample.shape = Shape.CIRCLE;
                        // sample.large_shape = Shape.LARGE_CIRCLE;
                    });
                }
            }
            if (add_pos_constraint) {
                for (let constraint_list of add_pos_constraint) {
                    constraint_list.forEach(idx => {
                        let sample = state.all_samples[idx];
                        sample.is_pos = true;
                        sample.is_neg = false;
                        // sample.shape = Shape.DOWN_TRIANGLE;
                        // sample.large_shape = Shape.LARGE_DOWN_TRIANGLE;
                        sample.color = Color.GREEN_POINT;
                    });
                }
            }      
            if (add_neg_constraint) {
                for (let constraint_list of add_neg_constraint) {
                    constraint_list.forEach(idx => {
                        let sample = state.all_samples[idx];
                        sample.is_pos = false;
                        sample.is_neg = true;
                        // sample.shape = Shape.DOWN_TRIANGLE;
                        // sample.large_shape = Shape.LARGE_DOWN_TRIANGLE;
                        sample.color = Color.RED_POINT;
                    });
                }
            }  
            if (add_up) {
                for (let constraint_list of add_up) {
                    constraint_list.forEach(idx => {
                        let sample = state.all_samples[idx];
                        sample.is_up = sample.is_pos = true;
                        sample.is_down = sample.is_neg = false;
                        sample.color = Color.GREEN_POINT;
                    });
                }
            }
            if (add_down) {
                for (let constraint_list of add_down) {
                    constraint_list.forEach(idx => {
                        let sample = state.all_samples[idx];
                        sample.is_up = sample.is_pos = false;
                        sample.is_down = sample.is_neg = true;
                        sample.color = Color.RED_POINT;
                    });
                }
            }
            if (reduce_reward) {
                for (let constraint_list of reduce_reward) {
                    constraint_list.forEach(idx => {
                        let sample = state.all_samples[idx];
                        sample.reduce_lam=true;
                    });
                }
            }
            if (enlarge_reward) {
                for (let constraint_list of enlarge_reward) {
                    constraint_list.forEach(idx => {
                        let sample = state.all_samples[idx];
                        sample.enlarge_lam=true;
                    });
                }
                console.log('tmp', state.all_samples.filter(d => d && d.enlarge_lam).map(d => d.row_i));
            }
        }
    },
    actions:{ // function to fetch data from backend
        async fetch_data({commit, state}){
            const resp = await axios.get(`${state.url}/data`,{'Access-Control-Allow-Origin': '*'});
            const data = JSON.parse(JSON.stringify(resp.data));
            commit('init_data', data);
            await __calculate_color(state);
        },
        async fetch_opt_result({commit, state}, payload) {
            state.constraint.pos_idx = state.all_samples.filter(d => d && d.is_pos).map(d => d.col_j);
            state.constraint.neg_idx = state.all_samples.filter(d => d && d.is_neg).map(d => d.col_j);
            state.constraint.up_idx = state.all_samples.filter(d => d && d.is_up).map(d => d.col_j);
            state.constraint.down_idx = state.all_samples.filter(d => d && d.is_down).map(d => d.col_j);
            state.all_samples.filter(d=>d&&d.col_x).forEach(d => d.is_up = d.is_down = false);
            state.constraint.reduce_idx = state.all_samples.filter(d => d && d.reduce_lam).map(d => d.row_i);
            state.constraint.enlarge_idx = state.all_samples.filter(d => d && d.enlarge_lam).map(d => d.row_i);
            state.all_samples.filter(d => d).forEach(d => d.reduce_lam=d.enlarge_lam=false);
            const resp = await axios.post(`${state.url}/update_weight`, state.constraint,
            {'Access-Control-Allow-Origin': '*'});
            const data = JSON.parse(JSON.stringify(resp.data));
            console.log(data);
            commit('set_opt_result', data);
            return data;
        },
        async fetch_sub_cluster({commit, state}, payload) {
            let cluster = payload.cluster;
            delete payload.cluster;
            const resp = await axios.post(`${state.url}/sub_cluster`, payload,
            {'Access-Control-Allow-Origin': '*'});
            const data = JSON.parse(JSON.stringify(resp.data));
            if (payload.type == 'row') {
                data.local_idx.forEach((idx, i) => {state.all_samples[state.row_index[idx]].row_sub_cluster_label = data.labels[i];})
            } else {
                data.local_idx.forEach((idx, i) => {state.all_samples[state.col_index[idx]].col_sub_cluster_label = data.labels[i];})
            }
            cluster.info.sub_cluster_label = data.labels;
            cluster.info.sub_cluster_count = Math.max.apply(null, data.labels) + 1;
        },
        async save({commit, state}, payload) {
            axios.post(`${state.url}/save_constraint`, {
                'corrected': state.all_samples.filter(d => d && d.corrected).map(d => d.index),
            });
        },

        async fetch_relationship({commit, state}, payload){
            // const resp = await axios.post(`${state.modelurl}/extract_relation`,{
            //     'row_index': payload.row_index,
            //     'col_index': state.col_index,
            // });
            // const new_R = JSON.parse(JSON.stringify(resp.data));
            const resp2 = await axios.post(`${state.url}/add_R`,{
                'row_index': payload.row_index,
                'row_label': payload.row_label,
            },{'Access-Control-Allow-Origin': '*'});
            const data = JSON.parse(JSON.stringify(resp2.data));
            await commit('update_R', {'data': data});
            return data;
        },



        async update_existing_relationship({commit, state}, payload){
            payload.row_index.forEach((idx,i) => {
                state.all_samples[idx].label = payload.row_label[i];
                state.all_samples[idx].correct = payload.row_label[i] == state.all_samples[idx].gt;
            })
            const resp2 = await axios.post(`${state.url}/replace_R`,{
                'row_index': payload.row_index,
                'row_label': payload.row_label,
            },{'Access-Control-Allow-Origin': '*'});
            const data = JSON.parse(JSON.stringify(resp2.data));
            await commit('replace_R', {'data': data});
            return data;
        },
    },
});

async function __calculate_color(state) {
    console.log('__calculate_color');
    if (state.color_type === 'groundtruth') state.colors = state.samples.map(d => d.gt).map(i => state.color_scale(i));
    else if (state.color_type === 'label') state.colors = state.samples.map(d => d.label).map(i => state.color_scale(i));
    else if (state.color_type === 'correct') state.colors = state.samples.map(d => d.correct ? "#00ff00" : "#ff0000");
    else state.colors = state.preds[state.epochs.indexof(state.color_type)].map(i => state.color_scale(i));
    state.scatter_colors = state.colors.map((color, i) => adjustHexColorOpacity(color, state.opacity[i]));
}

export default store;
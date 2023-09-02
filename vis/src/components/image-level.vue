<template>
    <v-col cols='12' class='fill-width' id='image-grid-column'>
    <div cols='12' class='topname '>Sample level</div>
    <div cols='12' class='h-line fill-width'></div>
    <svg :width='svgWidth' :height='svgHeight' :viewBox='`-15,-32,${viewWidth+15},${svgHeight}`' id='image-level'>
      <!-- <g transform="translate(0, -30)">
        <rect fill="#ffffff" x="-0" y="-0" :width="viewWidth"  height="20" rx="2" ry="2"></rect>
        <g v-for="(item) in legendData" :key="`legend-${item.label}`">
          <rect :fill="item.color" width=20 height=20 :x="item.x" :y="5"></rect>
          <text :x="item.x+25" :y="22.5">{{item.name}}</text>
        </g>
      </g> -->

      <g id="image-level-group" :transform="`translate(-0,0)`">
        <g v-if="items.length>0" transform="translate(-0,0)">
          <text :y=0 text-anchor="middle" dominant-baseline="hanging" :x='colWidth*0.5' font-size='1rem' >{{type==='train' ? 'Training' : 'Validation'}}</text>
          <text :y=0 text-anchor="middle" dominant-baseline="hanging" :x='colWidth*4.3' font-size='1rem' >{{type==='train' ? 'Positive validation samples' : 'Positive training samples'}}</text>
          <text :y=0 text-anchor="middle" dominant-baseline="hanging" :x='colWidth*10.2' font-size='1rem' >{{type==='train' ? 'Negative validation samples' : 'Negative training samples'}}</text>
        </g>
        <ImageLevelItem v-for="(d, i) in items" :key='`image-level-item-${d.idx}`' :i='i' :id='`row-group-${d.idx}`' :colWidth='colWidth' :rowHeight="rowHeight" :d=d></ImageLevelItem>
      </g>
    </svg>
    
    </v-col>
</template>

<script>
  import Vue from 'vue'
  import { select } from 'd3-selection';
  import { mapState, mapActions, mapMutations } from 'vuex'
  import { get_inner_width } from '@/plugins/utils.js';
  import { scaleLinear } from 'd3-scale';
  import * as reorder from 'reorder.js';

  const get_dist= (arr1, arr2) => {
        let cost = 0;
        for(let i = 0; i < arr1.length; ++i) {
            let index = arr2.indexOf(arr1[i]);
            if (index >= 0) cost += 1*Math.abs(i - index);
            else cost += 10;
        }
        return cost
    }
  const get_sys_dist = (arr1, arr2) => get_dist(arr1, arr2) + get_dist(arr2,arr1);


  export default Vue.extend({
    name: 'ImageGrid',
    components: {
        ImageLevelItem: () => import('./image-level-item.vue'),
    },
    mounted() {
      window.image_level = this;
      this.resize();
      window.addEventListener('resize', () => {
        this.redrawToggle = false;
        setTimeout(() => {
          this.redrawToggle = true;
          this.resize();
        }, 300);
      });
    },
    data() {
      return {
        svgWidth: 0,
        svgHeight: 1000,
        viewWidth: 590,
        redrawToggle: true,
        margin: {'left': 5, 'right': 5, 'top': 5, 'bottom': 5},
        imgMargin: 5,
        maxColCnt: 15,
        click_data: null,
        levelTranslate: 0,
      }
    },
    methods: {
      ...mapMutations(['update_color']),
      resize() {
        this.svgWidth = get_inner_width(document.getElementById('image-grid-column'));
      },
      col_reorder(col_idx) {
          let N = col_idx.length;
          let S = new Set(this.display_row_index);
          let distances = new Array(N).fill(null).map(_ => new Array(N).fill(0));
          let top5list = col_idx.map(idx => this.all_samples[idx].sorted_score_list.filter(idx => S.has(idx)).slice(S.size-5, S.size));
          let bottom5list = col_idx.map(idx => this.all_samples[idx].sorted_score_list.filter(idx => S.has(idx)).slice(0, 5));

          for (let i = 0; i < N; ++i) {
              for (let j = 0; j < N; ++j) {
                  distances[i][j] = distances[j][i] = 5 * get_sys_dist(top5list[i],top5list[j])  + get_sys_dist(bottom5list[i],bottom5list[j]);
              }
          }
        let rp = reorder.optimal_leaf_order().distanceMatrix(distances)(col_idx);
        return rp;
      }
    },
    computed: {
      ...mapState(['all_samples', 'highlight', 'highlight_row_index', 'highlight_col_index', 'matrix', 'lam', 'legend', 'R', 'row_index', 'col_index']),
      selected_val_index() {
        return this.matrix !== null ? this.matrix.row_idx : this.highlight_row_index;
      },
      selected_train_index() {
        return this.matrix !== null ? this.matrix.col_idx : this.highlight_col_index;
      },
      type() {
        if (this.selected_val_index.length == 0 && this.selected_train_index.length != 0) return 'train';
        if (this.selected_train_index.length == 0 && this.selected_val_index.length != 0) return 'val';
        return 'train';
      },
      items() {
        if (this.selected_val_index.length == 0 && this.selected_train_index.length == 0) {
          return [];
        }
        if (this.type === 'train') {
          let ret = null;
          if (this.selected_val_index.length == 0) {
            ret = this.selected_train_index.map(idx => ({
              'type': this.type,
              'idx': idx,
              'idxList': [idx],
              'neg': this.all_samples[idx].sorted_score_list.filter(_idx => this.R[this.row_index.indexOf(_idx)][this.col_index.indexOf(idx)]<=0).slice(0, 5),
              'pos': this.all_samples[idx].sorted_score_list.filter(_idx => this.R[this.row_index.indexOf(_idx)][this.col_index.indexOf(idx)]>=0).slice(-5).reverse(),
            }));
          } else {
            ret = this.selected_train_index.map(idx => ({
              'type': this.type,
              'idx': idx,
              'idxList': [idx],
              'neg': this.all_samples[idx].sorted_score_list.filter(_idx => this.R[this.row_index.indexOf(_idx)][this.col_index.indexOf(idx)]<=0 && this.selected_val_index.includes(_idx)).slice(0, 5),
              'pos': this.all_samples[idx].sorted_score_list.filter(_idx => this.R[this.row_index.indexOf(_idx)][this.col_index.indexOf(idx)]>=0 && this.selected_val_index.includes(_idx)).slice(-5).reverse(),
            }));
          }
          let sign = ret.map(d => this.all_samples[d.idx].score).sum() > 0 ? -1 : 1;
          console.log(ret, sign)
          ret.sort((a, b) => sign*(this.all_samples[a.idx].score - this.all_samples[b.idx].score));
          console.log(ret)
          return ret;
        } else {
          let ret = null;
          if (this.selected_train_index.length == 0) {
            ret = this.selected_val_index.map(idx => ({
              'type': this.type,
              'idx': idx,
              'idxList': [idx],
              'neg': this.all_samples[idx].sorted_influence_list.filter(_idx => this.R[this.row_index.indexOf(idx)][this.col_index.indexOf(_idx)]<=0).slice(0, 5),
              'pos': this.all_samples[idx].sorted_influence_list.filter(_idx => this.R[this.row_index.indexOf(idx)][this.col_index.indexOf(_idx)]>=0).slice(-5).reverse(),
            }));
          } else {
            ret = this.selected_train_index.map(idx => ({
              'type': this.type,
              'idx': idx,
              'idxList': [idx],
              'neg': this.all_samples[idx].sorted_influence_list.filter(_idx => this.R[this.row_index.indexOf(idx)][this.col_index.indexOf(_idx)]<=0 && this.selected_train_index.includes(_idx)).slice(0, 5),
              'pos': this.all_samples[idx].sorted_influence_list.filter(_idx => this.R[this.row_index.indexOf(idx)][this.col_index.indexOf(_idx)]<=0 && this.selected_train_index.includes(_idx)).slice(-5).reverse(),
            }));
          }
          ret.sort((a, b) => this.all_samples[b.idx].lam - this.all_samples[a.idx].lam);
          return ret;
        }
      },
      existing_idx() {
        return [this.selected_train_index, this.items.map(d => d['pos']).flat(), this.items.map(d => d['neg']).flat()].flat().unique();
      },

      colWidth() {
        return this.viewWidth / 13;
      },
      rowHeight() {
        return this.colWidth * 1.2;
      },
      legendData() {
          let ret = [];
          let acc_x = 0;
          for (let x of this.legend) {
              let w = x[1].length * 12;
              ret.push({
                  label: x[0],
                  name: x[1],
                  color: x[2],
                  width: w,
                  x: acc_x
              })
              acc_x += (w + 35);
          }
          let offset = (this.viewWidth - acc_x) / 2;
          ret.forEach(d => d.x += offset)
          return ret;
      }
    },
    watch: {
      existing_idx: function() {
        console.log(this.existing_idx)
        this.update_color(this.existing_idx.map(idx => this.all_samples[idx].label))
      },
      groupedIdx: function(_, oldValue) {
        // this.levelTranslate = (this.viewWidth - this.groupedIdx.length * this.colWidth) / 2;
        if (oldValue.length === 0) {
          // select("#image-level-group").attr("transform", `translate(${this.levelTranslate}, 0)`);
        }
        // select("#image-level-group")
        //   .transition()
        //   .duration(1000)
        //   .attr("transform", `translate(0, ${this.levelTranslate})`);
      }
    }
  });
</script>

<style scoped>
.image-level-transition-move {
  transition: transform 1s;
}
</style>
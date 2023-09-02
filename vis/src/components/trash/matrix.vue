<template>
    <v-col cols='12' class='fill-width' id='matrix-column'>
    <v-col cols='12' class='topname fill-width'>Image level</v-col>

    <svg v-if='redrawToggle === true' :width='svgWidth' :height='svgHeight' :viewBox='`${-margin.left},${-margin.top},${viewWidth+margin.left+margin.right},${viewHeight+margin.top+margin.bottom}`'>
   
        <g id="matrix-img-row" :transform="`translate(${0},${this.img_size[1] + 2})`">
            <ImageInfo v-for="(d, i) in rows" :key='`MatrixRow-${i}-ImageInfo-${d.idx}`' :parentId='`MatrixRow-${i}`' :idx="d.idx" :width='d.width' :height='d.height' :x='d.x' :y='d.y' :solid="d.solid"></ImageInfo>
        </g>
        <g id="matrix-img-col" :transform="`translate(${this.img_size[0] + 2},${0})`">
            <ImageInfo v-for="(d, i) in cols" :key='`MatrixCol-${i}-ImageInfo-${d.idx}`' :parentId='`MatrixCol-${i}`' :idx="d.idx" :width='d.width' :height='d.height' :x='d.x' :y='d.y'  :solid="d.solid"></ImageInfo>
        </g>
        <g id="matrix-cell" :transform="`translate(${this.img_size[0] + 2},${this.img_size[1] + 2})`">
            <rect v-for="d in cells" :key='`MatrixCell-${d.i}-${d.j}`' class="cell" :x="d.x" :y="d.y" :width="d.width" :height="d.height" :fill="d.color"></rect>
        </g>

    </svg>
    </v-col>
</template>

<script>
  import Vue from 'vue'
  
  import { mapState, mapMutations } from 'vuex'
  import { interpolate } from 'd3-interpolate';
  import { check_top_bottom, get_inner_width, Color } from '@/plugins/utils.js';


  export default Vue.extend({
    name: 'Graph',
    components: {
        ImageInfo: () => import('./image-info.vue'),
    },
    mounted() {
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
        redrawToggle: true,
        viewWidth: 500,
        viewHeight: 500,
        margin: {'left': 5, 'right': 5, 'top': 5, 'bottom': 5},
        mode: "image",
        pad_ratio: 0.9,
      }
    },
    methods: {
      ...mapMutations(['update_color']),
      resize() {
        this.svgWidth = get_inner_width(document.getElementById('matrix-column'));
      },
      weightColor(weight) {
        let polarity = weight > 0 ? this.GREEN : this.RED;
        let white = "#ffffff";
        return interpolate(white, polarity)(Math.abs(weight));
      },
    },
    computed: {
      ...mapState(['all_samples', 'row_index', 'col_index', 'num_col_cluster', 'num_row_cluster', 'lam', 'R', 'opt_result', 'highlight', 'high_pos_validation', 'high_neg_validation', 'matrix', 'color_map_list'
      ]),
      GREEN() {
          return Color.GREEN_POINT;
      },
      RED() {
          return Color.RED_POINT;
      },
      row_idx() {
          return this.matrix !== null ? this.matrix.row_idx : [...this.high_pos_validation.idx, ...this.high_neg_validation.idx].unique();
      },
      col_idx() {
          return this.matrix !== null ? this.matrix.col_idx : Array.from(this.highlight.idx);
      },
      img_size() {
          if (this.row_idx.length * this.col_idx.length===0) return [0, 0];
          let row_img_size, col_img_size;
          if (this.row_idx.length > this.col_idx.length) {
              row_img_size = this.viewHeight / (this.row_idx.length + this.row_idx.length / this.col_idx.length);
              col_img_size = this.viewHeight - row_img_size * this.row_idx.length;
          } else {
              col_img_size = this.viewWidth / (this.col_idx.length + this.col_idx.length / this.row_idx.length);
              row_img_size = this.viewWidth - col_img_size * this.col_idx.length;
          }
          return [Math.min(row_img_size, 40), Math.min(col_img_size, 40)];
      },
      sorted_row_idx() {
          return this.row_idx
          //.map(idx => ({idx, cnt: 0}))
            .map(idx => ({idx, cnt: this.col_idx.map(c => check_top_bottom(this.all_samples[c].sorted_score_list, idx, 15)).sum()}))
            //.filter(d => d.cnt > 0)
            //.map(idx => ({idx, cnt: this.high_pos_validation.count(idx)+this.high_neg_validation.count(idx)}))
            .sort((a,b) => b.cnt-a.cnt)
            //.slice(0, 40)
            .map(d => ({idx: d.idx, label: this.all_samples[d.idx].label}))
            .sort((a,b) => b.label-a.label)
            .map(d => d.idx);
      },
      sorted_col_idx() {
        return this.col_idx
            .map(idx => ({idx: idx, col_cluster: this.all_samples[idx].col_cluster}))
            .sort((a,b) => b.col_cluster-a.col_cluster)
            .map(d => d.idx);
      },
      rows() {
          if (!this.sorted_row_idx) return [];
          return this.sorted_row_idx.map((idx, i) => ({
            idx: idx,
            width: this.img_size[0] * this.pad_ratio,
            height: this.img_size[0] * this.pad_ratio,
            x: 0,
            y: (this.img_size[0] ) * i,
            strokeColor: this.color_map_list[this.all_samples[idx].label],
            solid: this.img_size[0] < 10,
          }))
      },
      cols() {
          if (!this.sorted_col_idx) return [];
          return this.sorted_col_idx.map((idx, j) => ({
            idx: idx,
            width: this.img_size[1] * this.pad_ratio,
            height: this.img_size[1] * this.pad_ratio,
            x: (this.img_size[1]) * j,
            y: 0,
            strokeColor: this.color_map_list[this.all_samples[idx].label],
            solid: this.img_size[1] < 10,
          }))
      },
      cells() {
          if (!this.sorted_col_idx || !this.sorted_row_idx) return [];
          let ret = [];
          this.sorted_row_idx.forEach((r, i) => {
            this.col_idx.forEach((c, j) => {
                let ii = this.row_index.indexOf(r);
                let jj = this.col_index.indexOf(c);
                ret.push({
                    i: i,
                    j: j,
                    x: j * (this.img_size[1]),
                    y: i * (this.img_size[0]),
                    width: this.img_size[1] * this.pad_ratio,
                    height: this.img_size[0] * this.pad_ratio,
                    color: this.weightColor(this.R[ii][jj]/50),
                })
            });
          });
          return ret;
      },
      svgHeight() {
        return this.svgWidth;
      },
    },
    watch: {
        matrix: function(para) {
            console.log('matrix', this.matrix);
            if (this.matrix) {
                this.update_color([...this.matrix.row_idx, ...this.matrix.col_idx].unique().map(idx => this.all_samples[idx].label))
            } else {
                this.update_color([...this.high_pos_validation.idx, ...this.high_neg_validation.idx, ...this.highlight.idx].unique().map(idx => this.all_samples[idx].label))
            }
            this.$parent.$forceUpdate();
        },
    }
  });
</script>

<style>

</style>
<template>
    <g :transform="`translate(0, ${y+5})`">

        <ImageInfo :parentId='`ListItem-${idx}`' :idx="idx" :width='imgSize' :height='imgSize' :x='0' :y='margin' ></ImageInfo>
        
        <text :x='imgSize/2' :y='margin+imgSize+margin+5' :font-size=10 text-anchor='middle' :fill='total_score<0?"red":"green"'>{{Math.round(total_score*1000)/1000}}</text>
        <g :transform="`translate(${imgSize+2.5}, 0)`">
          <text :x='0' :y='10' :font-size=10 text-anchor='start'>{{label_names[all_samples[idx].label]}}</text>
          <path v-if="all_samples[idx].is_pos || all_samples[idx].is_neg" :d="all_samples[idx].shape" :fill="all_samples[idx].color"
                :transform="`scale(1.5,1.5) translate(6,${imgSize/1.5})`" @click="toggleConstraint(idx)"></path>
        </g>
        <g v-for="(pair, i) in sorted_idx.slice(0, 5)" :key='`pos-score-${idx}-${pair.i}`' :transform="`translate(${imgSize+ 9*margin}, 0)`">
          <ImageInfo :parentId='`ListItem-${idx}`' :idx="row_index[pair.i]" :width='imgSize' :height='imgSize' :x='(imgSize + 2 * margin) * (i) ' :y='margin'></ImageInfo>
          <text :x='imgSize/2 + (imgSize + 2 * margin) * (i)' :y='margin+imgSize+margin+5' :font-size=10 text-anchor='middle'  :fill='pair.val<0?"red":"green"'>{{Math.round(pair.val*1000)/1000}}</text>
        </g>
        <g v-for="(pair, i) in sorted_idx.slice(sorted_idx.length-5).reverse()" :key='`neg-score-${idx}-${pair.i}`' :transform="`translate(${imgSize*6+ 22*margin}, 0)`">
          <ImageInfo :parentId='`ListItem-${idx}`' :idx="row_index[pair.i]" :width='imgSize' :height='imgSize' :x='(imgSize + 2 * margin) * (i) ' :y='margin' ></ImageInfo>
          <text :x='imgSize/2 + (imgSize + 2 * margin) * (i)' :y='margin+imgSize+margin+5' :font-size=10 text-anchor='middle'  :fill='pair.val<0?"red":"green"'>{{Math.round(pair.val*1000)/1000}}</text>
        </g>
    </g>
</template>

<script>
  import Vue from 'vue'
  import { select, selectAll } from 'd3-selection';
  import { mapState, mapActions, mapMutations } from 'vuex'
  import { get_inner_width } from '@/plugins/utils.js';
  import { scaleLinear } from 'd3-scale';
  import { extent } from 'd3-array';
  import { Shape, Color } from '@/plugins/utils.js';
  export default Vue.extend({
    name: 'ListItem',
    components: {
      ImageInfo: () => import('./image-info.vue'),
    },
    props: [
        'i',
        'idx',
        'y',
        'width',
        'height',
    ],
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
        redrawToggle: true,
      }
    },
    methods: {
      ...mapMutations(['set_update_cluster_by_idx']),
      resize() {
        this.svgWidth = get_inner_width(document.getElementById('image-grid-column'));
      },
      toggleConstraint(idx) {
        let sample = this.all_samples[idx];
        if (sample.is_pos) {
          sample.is_pos = false;
          sample.is_neg = true;
          // sample.shape = Shape.DOWN_TRIANGLE;
          // sample.large_shape = Shape.LARGE_DOWN_TRIANGLE;
          sample.color = Color.RED_POINT;
        } else if (sample.is_neg) {
          sample.is_pos = true;
          sample.is_neg = false;
          // sample.shape = Shape.UP_TRIANGLE;
          // sample.large_shape = Shape.LARGE_UP_TRIANGLE;
          sample.color = Color.GREEN_POINT;       
        }
        this.$forceUpdate();
        this.set_update_cluster_by_idx(idx);
      }
    },
    computed: {
      ...mapState(['all_samples', 'meta_data', 'R', 'lam', 'row_index', 'label_names']),
      id() {
        return `ListItem-${this.idx}`;
      },
      scores() {
        return this.lam.map((l, ii) => l * this.R[ii][this.all_samples[this.idx].col_j]);
      },
      total_score() {
        return this.scores.sum();
      },
      high_influence_idx() {
        return this.scores.map((val,i) => {return {val,i}}).sort((a,b)=>Math.abs(b.val)-Math.abs(a.val));
      },
      sorted_idx() {
        return this.scores.map((val,i) => {return {val,i}}).sort((a,b)=>b.val-a.val);
        //return this.lam.map((l, ii) => this.R[ii][this.all_samples[this.idx].col_j]).map((val,i) => {return {val,i}}).sort((a,b)=>b.val-a.val);
      },
      margin() {
          return this.height * 0.1;
      },
      imgSize() {
          return this.height * 0.8;
      },
    },
    watch: {
    }
  });
</script>


<style scoped>

</style>

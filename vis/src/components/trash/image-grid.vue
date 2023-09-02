<template>
    <v-col cols='12' class='fill-width' id='image-grid-column'>
    <v-col cols='12' class='topname fill-width'>Image List</v-col>
    <svg :width='svgWidth' :height='svgHeight' viewBox='-1,-1,602,602'>
      <text y='15' :font-size=15 text-anchor='middle' :x='imgSize/2'>Image</text>
      <text y='15' :font-size=15 text-anchor='middle' :x='imgSize+ 9*imgMargin + (imgSize + 2 * imgMargin) * 2.5'>Positive validation</text>
      <text y='15' :font-size=15 text-anchor='middle' :x='imgSize*6+ 22*imgMargin + (imgSize + 2 * imgMargin) * 2.5'>Negative validation</text>
      
      <ListItem v-for="(idx, i) in highlight_idx" :key='`image-${idx}`' :i='i' :idx='idx' :y='55 * i+12' :height='50' :width='500'></ListItem>
    </svg>
    </v-col>
</template>

<script>
  import Vue from 'vue'
  import { select } from 'd3-selection';
  import { mapState, mapActions, mapMutations } from 'vuex'
  import { get_inner_width } from '@/plugins/utils.js';
  import { scaleLinear } from 'd3-scale';

  export default Vue.extend({
    name: 'ImageGrid',
    components: {
        ListItem: () => import('./list-item.vue'),
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
        margin: {'left': 5, 'right': 5, 'top': 5, 'bottom': 5},
        imgSize: 40,
        imgMargin: 5,
      }
    },
    methods: {
      ...mapMutations(['update_color']),
      resize() {
        this.svgWidth = get_inner_width(document.getElementById('image-grid-column'));
      },
    },
    computed: {
      ...mapState(['all_samples', 'highlight', 'high_pos_validation', 'high_neg_validation', 'matrix']),
      col_idx() {
          return this.matrix !== null ? this.matrix.col_idx : this.highlight.idx;
      },
      highlight_idx() {
        return Array.from(this.col_idx).sort((a,b) => this.all_samples[a].score - this.all_samples[b].score);
      },
      svgHeight() {
        return this.svgWidth;
      },
    },
    watch: {
      highlight_idx: function() {
          if (this.matrix) {
              this.update_color([...this.matrix.row_idx, ...this.matrix.col_idx].unique().map(idx => this.all_samples[idx].label))
          } else {
              this.update_color([...this.high_pos_validation.idx, ...this.high_neg_validation.idx, ...this.highlight.idx].unique().map(idx => this.all_samples[idx].label))
          }
      }
    }
  });
</script>

<style scoped>

</style>
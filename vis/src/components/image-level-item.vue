<template>
    <g :transform="`translate(${0}, ${0})`" :id="id">
        <g :transform="`translate(${0}, ${0})`" @mouseover="mouseover(d.idxList, false)" @mouseout="mouseout(d.idxList, false)">
           <ImageInfo :parentId='id' :key="`${id}-image-{${d.idx}}`" :idx="d.idx" :width='imgSize' :height='imgSize' :x='0' :y='fontSize' :is_val="false" @click="mouseclick(d.idx, true)"></ImageInfo>
          <text :x='imgSize/2 ' :y='fontSize-2' :font-size='fontSize' text-anchor='middle'  >{{label_names[all_samples[d.idx].label]}}</text><text :x='imgSize/2' :y='imgSize+2*fontSize'  :font-size='fontSize' text-anchor='middle'  fill='black'>{{all_samples[d.idx].isTrain ? Math.round(val*1000*SCALE)/1000 : Math.round(val*1000)/1000}}</text>
          <path :d="all_samples[d.idx].shape" :fill="all_samples[d.idx].color" :transform="`translate(${(imgSize)-0.5*fontSize},${imgSize+1.6*fontSize}) scale(${fontSize/7},${fontSize/7})`" ></path>
        </g>
        <g v-for="(idx, i) in d.pos" :key='`pos-score-${idx}`' :transform="`translate(${(colMargin + colWidth) *(i+1) + groupMargin}, ${0})`" @mouseover="mouseover(idx, true)" @mouseout="mouseout(idx, true)" @click="mouseclick(idx, true)">
          <ImageInfo :parentId='id' :idx="idx" :width='imgSize' :height='imgSize' :x='0' :y='fontSize' :is_val="true"></ImageInfo>
          <text :x='imgSize/2 ' :y='fontSize-2' :font-size='fontSize' text-anchor='middle'  >{{label_names[all_samples[idx].label]}}</text><text :x='imgSize/2 ' :y='imgSize+2*fontSize' :font-size='fontSize' text-anchor='middle'  fill='trf'>{{Math.round(pos_val[i]*1000*SCALE)/1000}}</text>
          <path :d="all_samples[idx].shape" :fill="all_samples[idx].color" :transform="`translate(${(imgSize)-0.5*fontSize},${imgSize+1.6*fontSize}) scale(${fontSize/8},${fontSize/8})`"  ></path>
        </g>
        <g v-for="(idx, i) in d.neg" :key='`neg-score-${idx}`' :transform="`translate(${(colMargin + colWidth) *(i+1+topK) + groupMargin*2}, ${0})`"  @mouseover="mouseover(idx, true)" @mouseout="mouseout(idx, true)" @click="mouseclick(idx, true)">
          <!-- <rect fill="white" :x='0' :y='0' :width='imgSize' :height='rowHeight' stroke-width="1" stroke="black" ></rect> -->
          <ImageInfo :parentId='id' :idx="idx" :width='imgSize' :height='imgSize' :x='0' :y='fontSize' :is_val="true"></ImageInfo>
          <text :x='imgSize/2 ' :y='fontSize-2' :font-size='fontSize' text-anchor='middle'  >{{label_names[all_samples[idx].label]}}</text>
          <text :x='imgSize/2 ' :y='imgSize+2*fontSize' :font-size='fontSize' text-anchor='middle'  fill='black'>{{Math.round(pos_val[i]*1000)==0?'-':''}}{{Math.round(neg_val[i]*1000*SCALE)/1000}}</text> <!--:fill='neg_val[i]<0?"red":"green"'-->
          <path :d="all_samples[idx].shape" :fill="all_samples[idx].color" :transform="`translate(${(imgSize)-0.5*fontSize},${imgSize+1.6*fontSize}) scale(${fontSize/8},${fontSize/8})`"  ></path>
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
    name: 'ImageLevelItem',
    components: {
      ImageInfo: () => import('./image-info.vue'),
    },
    props: [
        'd',
        'i',
        'id',
        'colWidth',
        'cellSize',
        'rowHeight',
    ],
    mounted() {
      this.resize();
      select(`#${this.id}`)
        .attr("transform", `translate(${0}, ${this.i * (this.rowHeight + this.rowMargin) + 25})`)
        .style("opacity", 0)
        .transition()
        .duration(1000)
        .style("opacity", 1);
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
        topK: 5,
        SCALE:1,
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
      },
      highlight(idx, is_val) {
          this.node.selectAll(`.image-rect-${is_val?'row':'col'}-${idx}`).attr("stroke", Color.HIGHLIGHT);
          select(`#point-${idx}`).attr("fill", Color.HIGHLIGHT);
          // select(`#${is_val?'validation':'point'}-${idx}`).style("opacity", 1);
          // if (is_val) {
          //   const rect = select(`#validation-${idx} rect.point`);
          //   const y = +rect.attr('y')+rect.attr('height')/2
          //   rect.attr('x',-4).attr('width',8).attr('y', y-8).attr('height', 16).attr("fill",  Color.HIGHLIGHT);
          // }
      },
      dehighlight(idx, is_val) {
        this.node.selectAll(`.image-rect-${is_val?'row':'col'}-${idx}`).attr("stroke", this.color_map_list[this.all_samples[idx].label]);
        select(`#point-${idx}`).attr("fill", d => d.color);
        // select(`#${is_val?'validation':'point'}-${idx}`).style("opacity", 0.6);
        // if (is_val) {
        //   const rect = select(`#validation-${idx} rect.point`);
        //   const y = +rect.attr('y')+rect.attr('height')/2
        //   rect.attr('x',-2.5).attr('width',5).attr('y', y-5).attr('height', 10).attr("fill", this.all_samples[idx].color);
        //   let cnt1 = this.high_pos_validation.count(idx);
        //   let cnt2 = this.high_neg_validation.count(idx);
        //   if (cnt1 == 0 && cnt2 == 0) rect.attr('fill', "#666666").style("opacity", 0.6);
        //   else rect.attr('fill', cnt1 > cnt2 ? Color.GREEN_POINT : cnt1 < cnt2 ? Color.RED_POINT : "#666666").style("opacity", 1);
        // }
      },
      mouseover(idx, is_val) {
        console.log('mouseover', idx, is_val, this.$parent.click_data);
        if (this.$parent.click_data !== null && !(this.$parent.click_data[0]==idx && this.$parent.click_data[1]==is_val)) this.dehighlight(this.$parent.click_data[0], this.$parent.click_data[1])
        let idxList = (typeof idx === 'number') ? [idx] : idx;
        for (let idx of idxList) this.highlight(idx, is_val);
      },
      mouseout(idx, is_val) {
        console.log('mouseout', idx, is_val, this.$parent.click_data);
        let idxList = (typeof idx === 'number') ? [idx] : idx;
        for (let idx of idxList) this.dehighlight(idx, is_val);
        if (this.$parent.click_data !== null) this.highlight(this.$parent.click_data[0], this.$parent.click_data[1])
      },
      mouseclick(idx, is_val) {
        // console.log('mouseclick', idx, is_val, this.$parent.click_data);
        // if (this.$parent.click_data === null) {
        //   this.$parent.click_data = [idx, is_val];
        // } else {
        //   if (this.$parent.click_data[0] === idx && this.$parent.click_data[1] === is_val) {
        //     this.dehighlight(this.$parent.click_data[0], this.$parent.click_data[1]);
        //     this.$parent.click_data = null;
        //   } else {
        //     this.$parent.click_data = [idx, is_val];
        //   }
        // }
      }
    },
    computed: {
      ...mapState(['all_samples', 'meta_data', 'R', 'lam', 'row_index', 'color_map_list', 'high_pos_validation','high_neg_validation','consider_lam', 'row_index', 'col_index', 'R', 'label_names']),
      val() {
        if (this.d.type == 'val') {
          return this.all_samples[this.d.idx].lam;
        } else {
          return this.all_samples[this.d.idx].score;
        }
      },
      neg_val() {
        if (this.d.type == 'val') {
          let r = this.row_index.indexOf(this.d.idx);
          return this.d.neg.map(idx => this.all_samples[this.d.idx].lam * this.R[r][this.col_index.indexOf(idx)]);
        } else {
          let c = this.col_index.indexOf(this.d.idx);
          return this.d.neg.map(idx => this.all_samples[idx].lam * this.R[this.row_index.indexOf(idx)][c]);
        }
      },
      pos_val() {
        if (this.d.type == 'val') {
          let r = this.row_index.indexOf(this.d.idx);
          return this.d.pos.map(idx => this.all_samples[this.d.idx].lam * this.R[r][this.col_index.indexOf(idx)]);
        } else {
          let c = this.col_index.indexOf(this.d.idx);
          return this.d.pos.map(idx => this.all_samples[idx].lam * this.R[this.row_index.indexOf(idx)][c]);
        }
      },
      node(){
        return select(`#image-level`);
      },
      colMargin() {
         return this.colWidth * 0.1;
      },
      rowMargin() {
        return this.rowHeight * 0.4;
      },
      imgSize() {
        return this.colWidth;
        //return this.cellSize * 0.8;
      },
      fontSize() {
          return (this.rowHeight-this.colWidth);
      },
      groupMargin() {
        // return this.rowHeight / 10;
        return this.colWidth * 0.45;
      },
    },
    watch: {
      i: function() {
        select(`#${this.id}`)
          .transition()
          .duration(1000)
          .attr("transform", `translate(${0}, ${this.i * (this.rowHeight + this.rowMargin) + 25})`)
      }
    }
  });
</script>


<style scoped>

</style>

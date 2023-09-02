<template>
    <g :transform="`translate(${0}, ${0})`" :id="id">
        <g :transform="`translate(${colMargin+imgOffset}, ${0})`" @mouseover="mouseover(idxList, false)" @mouseout="mouseout(idxList, false)">
          <ImageInfo v-for="(idx, i) in idxStackList" :parentId='id' :key="`${id}-image-{${idx}}`" :idx="idx" :width='imgSize' :height='rowHeight-2*imgOffset' :x='0+2*(idxStackList.length-1-i)' :y='imgOffset-2*(idxStackList.length-1-i)' :is_val="false" @click="mouseclick(row_index[pair.i], true)"></ImageInfo>
          <!--
          <ImageInfo :parentId='id' :key="`${id}-image-{${idxList[0]}}`" :idx="idxList[0]" :width='imgSize' :height='rowHeight-2*imgOffset' :x='0' :y='imgOffset' :is_val="false" @click="mouseclick(row_index[pair.i], true)"></ImageInfo>
          -->
          <circle v-show="idxList.length > 1" :cx="imgSize" :cy="imgOffset" :r="(cellSize-imgSize)/2+1" stroke="black" fill="white"></circle>
          <text v-show="idxList.length > 1" :x="imgSize" :y="imgOffset" fill="black" text-anchor="middle" dominant-baseline="middle" font-size="12">{{idxList.length}}</text>
          <rect fill="white" :x='0' :y='imgOffset+imgSize' :width='imgSize' :height='fontSize'></rect>
          <text :x='colMargin ' :y='imgOffset+imgSize+fontSize-1' :font-size='fontSize' text-anchor='start'  :fill='total_score<0?"red":"green"'>{{Math.round(total_score*1000)/1000}}</text>
        
          <path v-if="idxList.map(idx=>all_samples[idx].is_pos).every(d=>d) || idxList.map(idx=>all_samples[idx].is_neg).every(d=>d)" :d="all_samples[IDX].shape" :fill="all_samples[IDX].color"
                :transform="`translate(${(imgSize)-6},${imgOffset+imgSize+fontSize-(all_samples[IDX].is_pos?4:6)}) scale(${fontSize/7},${fontSize/7})`  " @click="toggleConstraint(IDX)" ></path>
        </g>
        <!-- <g v-for="(pair, i) in topIdx" :key='`pos-score-${pair.i}`' :transform="`translate(${colMargin+imgOffset}, ${rowHeight *(i+1) + groupMargin})`" @mouseover="mouseover(row_index[pair.i], true)" @mouseout="mouseout(row_index[pair.i], true)" @click="mouseclick(row_index[pair.i], true)"> -->
        <g v-for="(pair, i) in topIdx" :key='`pos-score-${pair.i}`' :transform="`translate(${colMargin+imgOffset + colWidth *(i+1) + groupMargin}, ${0})`" @mouseover="mouseover(row_index[pair.i], true)" @mouseout="mouseout(row_index[pair.i], true)" @click="mouseclick(row_index[pair.i], true)">
          <ImageInfo :parentId='id' :idx="row_index[pair.i]" :width='imgSize' :height='rowHeight-2*imgOffset' :x='0' :y='imgOffset' :is_val="true"></ImageInfo>
          <rect fill="white" :x='0' :y='imgOffset+imgSize' :width='imgSize' :height='fontSize'></rect>
          <text :x='imgSize/2 ' :y='imgOffset+imgSize+fontSize-1' :font-size='fontSize' text-anchor='middle'  :fill='pair.val<0?"red":"green"'>{{Math.round(pair.val*1000)/1000}}</text>
        </g>
        <!-- <g v-for="(pair, i) in bottomIdx" :key='`neg-score-${pair.i}`' :transform="`translate(${colMargin+imgOffset}, ${rowHeight *(i+6) + groupMargin*2})`"  @mouseover="mouseover(row_index[pair.i], true)" @mouseout="mouseout(row_index[pair.i], true)" @click="mouseclick(row_index[pair.i], true)"> -->
        <g v-for="(pair, i) in bottomIdx" :key='`neg-score-${pair.i}`' :transform="`translate(${colMargin+imgOffset + colWidth *(i+1+topK) + groupMargin*2}, ${0})`"  @mouseover="mouseover(row_index[pair.i], true)" @mouseout="mouseout(row_index[pair.i], true)" @click="mouseclick(row_index[pair.i], true)">
          <ImageInfo :parentId='id' :idx="row_index[pair.i]" :width='imgSize' :height='rowHeight-2*imgOffset' :x='0' :y='imgOffset' :is_val="true"></ImageInfo>
          <rect fill="white" :x='0' :y='imgOffset+imgSize' :width='imgSize' :height='fontSize'></rect>
          <text :x='imgSize/2 ' :y='imgOffset+imgSize+fontSize-1' :font-size='fontSize' text-anchor='middle'  :fill='pair.val<0?"red":"green"'>{{Math.round(pair.val*1000)/1000}}</text>
        </g>
        <!--
        <g v-for="(pair, i) in otherIdx" :key='`other-score-${pair.i}`' :transform="`translate(${colMargin+imgOffset}, ${rowHeight *(i+11) + groupMargin*3})`" @mouseover="mouseover(row_index[pair.i], true)" @mouseout="mouseout(row_index[pair.i], true)" @click="mouseclick(row_index[pair.i], true)">
          <ImageInfo :parentId='id' :idx="row_index[pair.i]" :width='imgSize' :height='rowHeight-2*imgOffset' :x='0' :y='imgOffset' :is_val="true"></ImageInfo>
          <rect fill="white" :x='0' :y='imgOffset+imgSize' :width='imgSize' :height='fontSize'></rect>
          <text :x='imgSize/2 ' :y='imgOffset+imgSize+fontSize-1' :font-size='fontSize' text-anchor='middle'  :fill='pair.val<0?"red":"green"'>{{Math.round(pair.val*1000)/1000}}</text>
        </g>
        -->
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
        'i',
        'id',
        'idxList',
        'colWidth',
        'cellSize',
        'display_row_index',
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
          // select(`#${is_val?'validation':'point'}-${idx}`).style("opacity", 1);
          // if (is_val) {
          //   const rect = select(`#validation-${idx} rect.point`);
          //   const y = +rect.attr('y')+rect.attr('height')/2
          //   rect.attr('x',-4).attr('width',8).attr('y', y-8).attr('height', 16).attr("fill",  Color.HIGHLIGHT);
          // }
      },
      dehighlight(idx, is_val) {
        this.node.selectAll(`.image-rect-${is_val?'row':'col'}-${idx}`).attr("stroke", this.color_map_list[this.all_samples[idx].label]);
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
        console.log('mouseclick', idx, is_val, this.$parent.click_data);
        if (this.$parent.click_data === null) {
          this.$parent.click_data = [idx, is_val];
        } else {
          if (this.$parent.click_data[0] === idx && this.$parent.click_data[1] === is_val) {
            this.dehighlight(this.$parent.click_data[0], this.$parent.click_data[1]);
            this.$parent.click_data = null;
          } else {
            this.$parent.click_data = [idx, is_val];
          }
        }
      }
    },
    computed: {
      ...mapState(['all_samples', 'meta_data', 'R', 'lam', 'row_index', 'label_names', 'color_map_list', 'high_pos_validation','high_neg_validation','consider_lam']),
      idxStackList() {
        let ret = [];
        const thresholds = [1, 5];
        ret.push(this.idxList[0]);
        for (let thres of thresholds) {
          if (this.idxList.length > thres) ret.push(this.idxList[thres]);
        }
        return ret;
      },
      node(){
        return select(`#image-level`);
      },
      IDX() {
          return this.idxList[0];
      },
      rowI() {
          return new Set(this.display_row_index.map(idx => this.all_samples[idx].row_i));
      },
      scores() {
        return this.consider_lam ? this.idxList.map(idx => this.all_samples[idx].score_list).mean2()
          : this.idxList.map(idx => this.all_samples[idx].raw_score_list).mean2();
        // return this.lam.map((l, ii) => this.idxList.map(idx => l * this.R[ii][this.all_samples[idx].col_j]).mean());
        // return this.display_row_index.map((rIdx) => this.idxList.map(cIdx => this.lam[this.all_samples[rIdx].row_i] * this.R[this.all_samples[rIdx].row_i][this.all_samples[cIdx].col_j]).mean());
      },
      total_score() {
        return this.consider_lam ? this.idxList.map(idx => this.all_samples[idx].score).mean()
          : this.idxList.map(idx => this.all_samples[idx].raw_score).mean()  ;
      },
      sorted_idx() {
        return this.scores.map((val,i) => {return {val,i}}).sort((a,b)=>b.val-a.val).filter(d => this.rowI.has(d.i));
        //return this.lam.map((l, ii) => this.R[ii][this.all_samples[this.idx].col_j]).map((val,i) => {return {val,i}}).sort((a,b)=>b.val-a.val);
      },
      topIdx() {
        return this.sorted_idx.slice(0, this.topK).filter(d => d.val > 0);
      },
      bottomIdx() {
        return this.sorted_idx.slice(this.sorted_idx.length-this.topK).filter(d => d.val < 0).reverse();
      },
      otherIdx() {
        return this.sorted_idx.slice(this.topIdx.length, this.sorted_idx.length-this.bottomIdx.length);
      },
      colMargin() {
          return (this.colWidth - this.cellSize) / 2;
      },
      rowMargin() {
        return this.rowHeight * 0.1;
      },
      imgSize() {
        return this.cellSize * 0.75;
      },
      imgOffset() {
        return (this.cellSize - this.imgSize) / 2;
      },
      fontSize() {
          return this.rowHeight-this.cellSize;
      },
      groupMargin() {
        // return this.rowHeight / 10;
        return this.colWidth / 4;
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

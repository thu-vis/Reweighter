<template>
<v-col cols="12">
<v-row cols='12' class='topname fill-width'>Config</v-row>
<v-row justify="center" align="center">
    <v-subheader>Color: </v-subheader>
    <v-select
          v-model="color_type_select"
          :items="color_items"
          single-line
        ></v-select>
</v-row>
<v-row justify="center" align="center">
    <v-subheader>Opacity: </v-subheader>
    <v-autocomplete
          v-model="color_opacity_select"
          :items="opacity_items"
          single-line
        ></v-autocomplete>
</v-row>

<v-row justify="center" align="center">
<v-subheader>Attributes: </v-subheader>
<v-autocomplete
      v-model="pcp_attributes"
      :items="opacity_items"
      single-line
      multiple
    >
      <template v-slot:selection="{ item, index }">
        <v-chip v-if="index <= 0">
          <span>{{ shorten(item) }}</span>
        </v-chip>
        <!--<span v-if="index === 2" class="grey--text text-caption" >
          (+{{ select_attributes.length - 2 }})
        </span>-->
      </template>
  </v-autocomplete>
</v-row>

<v-row justify="center" align="center">
    <v-subheader>Line: </v-subheader>
    <v-select
          v-model="line_attribute"
          :items="line_items"
          single-line
        ></v-select>
</v-row>

</v-col>
</template>

<script>

  import Vue from 'vue';
  import { mapState, mapMutations } from 'vuex'
  import { attr_shorten } from '@/plugins/utils.js'
  export default Vue.extend({
    name: 'Config',
    mounted() {
    },
    data () {
      return {
        color_type_select: 'groundtruth',
        color_items: [
          'groundtruth',
          'label', 
          'pred',
          'correct',  
        ],
        color_opacity_select: 'uniform',
        select_attributes:[],
        line_items: [
          'none',
          'weight',
          'margin',
          //'consistency',
        ],
      }
    },
    methods: {
      ...mapMutations(['set_color_type', 'set_color_opacity', 'set_pcp_attributes', 'set_line_attribute']),
      shorten(attr) {
        return attr_shorten(attr)
      },
    },
    computed: {
      ...mapState(['meta_data']),
      opacity_items() {
        if (!this.meta_data.valid_epochs) return ['uniform'];
        return ['uniform'].concat(this.meta_data.valid_epochs.map(epoch => `weight-epoch${epoch}`))
        .concat(this.meta_data.valid_epochs.map(epoch => `margin-epoch${epoch}`));
        //.concat(this.meta_data.valid_epochs.map(epoch => `consistency-epoch${epoch}`));
      },
      pcp_attributes: {
        get () { return this.$store.state.pcp_attributes},
        set (value) { this.set_pcp_attributes(value) }
      },
      line_attribute: {
        get () { return this.$store.state.line_attribute},
        set (value) { this.set_line_attribute(value) }
      }
    },
    watch: {
      color_opacity_select(val) {
        this.set_color_opacity(val);
      },
      color_type_select(val) {
        this.set_color_type(val);
      },
    }
  });
</script>

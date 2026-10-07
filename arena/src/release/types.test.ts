import { describe, test, expect } from 'bun:test';
import { selectedTask, validateRelease } from './types';
import type { Release } from './types';

function release(): Release {
  return {schemaVersion: 1, version:'test', state:'draft', datasets:[], selectionSha256:'x', sourceSha256:{},
    tasks:['marker_d2','square_d2','routing_d2','square_narrow','square_broad'].map(id => ({id,title:id,domain:'real',datasetTask:id,metric:'full_success',policies:[],blocks:[]}))};
}
describe('release membership', () => {
  test('unknown task never falls back to live data', () => {
    expect(() => selectedTask(release(), 'square_fixed_peg')).toThrow('not part of the release');
  });
  test('rejects extra and missing tasks', () => {
    const r=release();r.tasks.pop();expect(() => validateRelease(r)).toThrow('Incomplete task');
  });
  test('rejects a block outside the dataset allowlist', () => {
    const r=release();r.tasks[0].blocks.push({id:'excluded',round:'R0',dataset:'example/pilot',source:'example/pilot',revision:'a'.repeat(40),fps:15,cameras:[],reviewedEpisodes:null,sourcePath:'x',starts:[]});
    expect(() => validateRelease(r)).toThrow('Unpinned eval block');
  });
  test('rejects an excluded opponent inside a mainline recording', () => {
    const r=release();r.datasets=[{id:'mulligan/mainline',task:'marker_d2',role:'evaluation',variant:null,episodes:1,frames:1,fps:15,cameras:[],parent:null,source:'example/source',revision:'a'.repeat(40),tier:'mainline'}];
    r.tasks[0].blocks.push({id:'block',round:'R0',dataset:'mulligan/mainline',source:'example/source',revision:'a'.repeat(40),fps:15,cameras:[],reviewedEpisodes:null,sourcePath:'x',starts:[{index:0,results:[{policyId:'excluded',episode:0,success:false,outcome:'failure',steps:1,frames:1,score:0,marks:null,videos:{}}]}]});
    expect(() => validateRelease(r)).toThrow('Unselected policy');
  });
});

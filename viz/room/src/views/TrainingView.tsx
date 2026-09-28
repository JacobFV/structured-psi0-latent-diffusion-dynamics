/** Training (IBM-2's third view), with ops folded in: peer vitals, leases, DAG progress and broker events as charts, then
 * the training curves and gradient health. Tables only behind "show data tables". */
import { SideGroup, SidebarControls } from '../components/shell';
import { useUrlState } from '../lib/url';
import type { LensDef } from './Lens';
import LiveOps from './LiveOps';
import OpsCharts from './OpsCharts';
import Training from './Training';

export const TRAINING: LensDef[] = [];
export default function TrainingView() {
  const [data, setData] = useUrlState('data', '0');
  return (
    <>
      <SidebarControls>
        <SideGroup title="Tables">
          <label className="check"><input type="checkbox" checked={data === '1'} onChange={(e) => setData(e.target.checked ? '1' : '0')} /> show data tables</label>
        </SideGroup>
      </SidebarControls>
      <OpsCharts />
      <Training />
      {data === '1' && <><h2 className="sect">Ops tables</h2><LiveOps /></>}
    </>
  );
}

import {
  getClusteringResults,
  getRepresentationAnalysis,
  getConfusedPairs,
  getNearestNeighborResults,
  getUmapPoints,
  getFinalReport,
} from '@/lib/data';
import ResearchDashboard from './ResearchDashboard';

export default function Home() {
  const clusteringResults = getClusteringResults();
  const representationAnalysis = getRepresentationAnalysis();
  const confusedPairs = getConfusedPairs();
  const nearestNeighborResults = getNearestNeighborResults();
  const umapPoints = getUmapPoints();
  const finalReport = getFinalReport();

  return (
    <ResearchDashboard
      clusteringResults={clusteringResults}
      representationAnalysis={representationAnalysis}
      confusedPairs={confusedPairs}
      nearestNeighborResults={nearestNeighborResults}
      umapPoints={umapPoints}
      finalReport={finalReport}
    />
  );
}
